"""低置信度信号（路线图 M3.4）：判断一次检索结果「值不值得补救」。

要解决什么问题
--------------
主链路（向量 + BM25 + RRF + 重排）对大多数问题是够用的，但总有一类问题检索完
之后结果明显可疑：

    - 前两名分数咬得很紧，说明重排自己也拿不准哪个才对；
    - 问题里的关键词（人名、功法名）在候选原文里根本没出现，
      说明召回的大概率只是「主题相近」，不是「答案」；
    - 候选全挤在同一本书里且分数接近，可能是「庄主」「师父」这类
      实体指代歧义——用户想问的书和我们猜的不是同一本。

这些情况与其让生成模型硬编，不如识别出来后做**一次**有边界的补救
（自适应查询扩展，见 ``query_expander.py`` 和 ``rag.py`` 的挂钩点）。

为什么只用重排器分数
--------------------
向量余弦相似度和 BM25 分数量纲完全不同（前者约 0~1，后者可以到几十），
把它们混在一起定统一阈值必然顾此失彼——这是路线图明确禁止的。交叉编码器的
原始输出是任意范围的 logit，先用 sigmoid 压到 (0,1) 再比差距，所有候选
用的是同一把尺子。

设计边界
--------
- **纯函数**：不碰数据库、不加载模型，输入打分好的候选列表即可离线复现，
  这样 ``scripts/eval_low_confidence.py`` 才能用同一份代码做阈值校准。
- **只给信号，不替人做决定**：阈值全部是经验起点，最终取值必须由离线评测
  数据说了算（见脚本用法），不能拍脑袋上调完就直接影响线上行为。
"""

import math

from chunk_model import SourceChunk
from tokenizer import query_terms


def normalized_score(raw: float) -> float:
    """把重排器的原始 logit 压到 (0,1)，数值上稳定的 sigmoid。

    不直接用 1/(1+exp(-x)) 是因为 logit 可能很极端（bge-reranker 实测能到
    ±20），exp(-(-20)) 会直接溢出。按符号拆成两条路径，指数参数永远 ≤ 0。
    """
    x = float(raw)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --- 阈值：全部是经验起点，待离线校准 -------------------------------------
#
# **当前取值没有任何实测依据支撑**，只是按「合理量级」给的占位值：
# 正确答案被重排顶到第一时，与第二名的归一化分差通常很明显；而两个都半懂
# 不懂的候选往往挤在一起。分界线画在哪，必须跑
#     python scripts/eval_low_confidence.py
# 看「低置信组的命中率是否显著低于正常组」再定。在那之前不要调大触发范围。
SCORE_GAP_LOW = 0.05
# 问题关键词覆盖率低于这个值，认为候选文本基本没接住问题。
TERM_COVERAGE_MIN = 0.5


def compute_confidence(
    question: str,
    scored_candidates: list[tuple[SourceChunk, float]],
) -> dict:
    """从重排后的候选计算低置信度信号。

    参数
    ----
    question          用户原始问题（不是改写后的检索查询）
    scored_candidates [(SourceChunk, 重排器原始分)]，**按分数从高到低排好序**
                      ——就是 ``rerank_with_scores`` 直接吐出来的形状。

    返回字典字段：
        score_gap             第 1 名与第 2 名的归一化分差；只有 1 条候选时
                              记 1.0（无从比较 ≠ 有歧义，不该触发补救）
        term_coverage         问题分词后在候选文本中的覆盖率
        cross_book_dispersion 候选跨越几本书
        low_signals           触发了哪些信号（空列表 = 置信度正常）
        is_low_confidence     综合判定

    判定规则刻意保守（默认少补救、绝不循环补救，见 rag.py 挂钩点）：
    - 覆盖率低单独触发——候选连问题词都没出现，几乎肯定没召回答案；
    - 分差小**还要叠加跨书 ≤ 1** 才触发——多书分散且分数接近更像问题本身
      模糊（问得太泛），换措辞重查帮助不大；而「都在一本书里还咬得紧」
      更像实体歧义（同名人物/称号），正是改写能救的场景。
    """
    chunks = [chunk for chunk, _raw in scored_candidates]

    # 信号一：第 1、2 名的归一化分差。只用重排分数，绝不掺向量/BM25 原始分。
    norm = [normalized_score(raw) for _chunk, raw in scored_candidates]
    if len(norm) >= 2:
        score_gap = norm[0] - norm[1]
    elif norm:
        score_gap = 1.0
    else:
        # 一条候选都没有属于「完全无证据」，走现有的无证据拒答逻辑；
        # 查询扩展救不了空结果（没有变体能从空索引里变出东西），不触发。
        return {
            "score_gap": 0.0,
            "term_coverage": 0.0,
            "cross_book_dispersion": 0,
            "low_signals": [],
            "is_low_confidence": False,
        }

    # 信号二：问题关键词覆盖率。和 BM25 用同一套分词，保证「覆盖」的口径
    # 与检索一致；纯停用词的问题（如"讲讲呗"）没有可检查的词，视为全覆盖。
    terms = query_terms(question)
    if terms:
        haystack = "\n".join(chunk.text for chunk in chunks)
        covered = sum(1 for term in terms if term in haystack)
        term_coverage = covered / len(terms)
    else:
        term_coverage = 1.0

    # 信号三：候选跨越几本书。0 或 1 本且分数接近 → 可能是实体歧义信号。
    cross_book_dispersion = len({chunk.novel for chunk in chunks})

    low_signals: list[str] = []
    if term_coverage < TERM_COVERAGE_MIN:
        low_signals.append("term_coverage")
    # 分差小本身不触发：多书分散且分数接近更像「问题问得太泛」，换措辞重查
    # 帮助不大；只有「分数接近且候选全挤在一本书里」（实体歧义特征）才补救。
    if score_gap < SCORE_GAP_LOW and cross_book_dispersion <= 1:
        low_signals.extend(["score_gap", "cross_book_dispersion"])

    return {
        "score_gap": round(score_gap, 6),
        "term_coverage": round(term_coverage, 6),
        "cross_book_dispersion": cross_book_dispersion,
        "low_signals": low_signals,
        "is_low_confidence": bool(low_signals),
    }
