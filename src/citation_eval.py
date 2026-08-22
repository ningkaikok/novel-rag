"""引用的离线可验证指标（M3.5 起，引用质量拆成三个可分别计算的维度）。

三个维度各自回答一个问题，互不可替代：

- **正确性**：回答里的 ``[n]`` 编号是否指向真实来源？——``valid_number_ratio``
  （纯规则，可在线统计）
- **完整性**：回答中的"事实性陈述"是否带引用支撑？——``completeness`` 分组
  （按句末标点分句后统计不含任何引用编号的陈述占比；寒暄/元描述句按豁免词表
  跳过。纯规则，可在线统计）
- **忠实度**：被引用的原文是否真的支持该陈述？——``judge_support`` 影子接口
  （需要语义判断，只能靠 LLM Judge 或人工标注；**只在评测脚本里调用，
  绝不进入问答主链路**，见 docs/roadmap.md「M3.5」）

纯规则永远不能证明语义蕴含，所以返回值明确保留 ``manual_support_review_required``，
避免把"编号合法"误写成"事实一定被支持"。
"""
import json
import re
from collections.abc import Mapping, Sequence

_CITATION_RE = re.compile(r"\[(\d+)]")

# 句子切分：在句号/问号/叹号（含半角）之后断开。分号、逗号不断——中文小说里
# 一个长句内部常用分号并列多个动作，拆开会让"陈述"粒度过碎。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。？！?!])\s*")

# 完整性豁免词表（**启发式**，仅用于离线评测统计，不影响线上任何行为）。
# 这些句式不承载"来自原文的小说事实"，没有 [n] 引用不算完整性缺陷：
# - 元描述句：描述的是"本次回答是怎么来的"，不是小说内容；
# - 拒答句：模型如实说"证据不足"，本来就不该有引用；
# - 寒暄/收尾客套：与事实核验无关。
# 词表刻意用较长的短语而不是单字，降低把正常事实句误豁免的概率；
# 允许调用方传入自定义列表整体替换（evaluate_citations 的 exempt_phrases 参数）。
DEFAULT_EXEMPT_PHRASES: tuple[str, ...] = (
    # --- 元描述 ---
    "根据检索结果",
    "根据提供的片段",
    "根据以上片段",
    "根据原文",
    "综合来看",
    "从片段来看",
    "以下是",
    "整理如下",
    # --- 拒答 ---
    "无法确定",
    "无法回答",
    "没有提到",
    "未提及",
    "不足以",
    # --- 寒暄 / 收尾客套 ---
    "希望对你有帮助",
    "希望这个回答",
    "很高兴为你",
    "祝你阅读愉快",
    "欢迎继续提问",
    "如果还有其他问题",
)

# ---------------------------------------------------------------- 忠实度影子接口

_JUDGE_PROMPT = """你是一个严格的核验员。请判断下面的【陈述】能否从【证据】中直接推出
（允许同义改写和代词指代，但不允许添加证据中没有的事实、数字或因果关系）。

【陈述】
{statement}

【证据】
{evidence}

注意：证据是小说原文，其中的比喻、人物绰号和代词指代应视为有效支撑；
但时间顺序颠倒、数字篡改、张冠李戴都属于不支持。

只输出一个 JSON 对象（不要输出其他任何文字）：
{{"label": "supported|unsupported|uncertain", "reason": "一句话理由"}}"""


def _strip_code_fence(text: str) -> str:
    """剥掉模型输出常见的 markdown 代码围栏，取出中间内容。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉首行 ```json / ``` 之类的围栏标记和结尾 ```
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _extract_json_object(text: str) -> dict | None:
    """从模型输出里提取第一个 JSON 对象；解析失败返回 None。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def judge_support(
    statement: str,
    evidence_texts: Sequence[str],
    generate_fn,
    errors: list | None = None,
) -> dict:
    """忠实度影子接口：让 LLM Judge 判断证据是否支持一条陈述。

    ``generate_fn`` 是注入的生成函数（prompt -> token 迭代器），复用
    query_rewriter/query_expander 的依赖注入模式：本模块不依赖任何云端 SDK，
    Web 主链路也**绝不**调用本函数——它只服务于评测脚本（如
    scripts/eval_faithfulness_shadow.py）和测试。

    返回 ``{"label": "supported|unsupported|uncertain", "reason": str}``。
    任何失败（生成异常、输出不是 JSON、label 非法）都降级为
    ``{"label": "uncertain", ...}`` 并把原因追加进 errors（如果给了）——
    影子评测宁可记录"不确定"也不能编造一个确定标签。
    """
    evidence = "\n".join(f"- {text}" for text in evidence_texts)
    prompt = _JUDGE_PROMPT.format(statement=statement, evidence=evidence)
    try:
        raw = "".join(generate_fn(prompt)).strip()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if errors is not None:
            errors.append(reason)
        return {"label": "uncertain", "reason": f"Judge 调用失败：{reason}"}
    try:
        data = _extract_json_object(_strip_code_fence(raw))
        if data is None:
            raise ValueError("输出中找不到 JSON 对象")
        label = str(data.get("label", "")).strip().lower()
        if label not in ("supported", "unsupported", "uncertain"):
            raise ValueError(f"label 非法：{label!r}")
    except Exception as exc:
        reason = (
            f"{type(exc).__name__}: {exc}；原始输出前 200 字："
            + raw[:200].replace("\n", "\\n")
        )
        if errors is not None:
            errors.append(reason)
        return {"label": "uncertain", "reason": f"Judge 输出解析失败：{reason}"}
    return {"label": label, "reason": str(data.get("reason", "")).strip()}


def rule_support(statement: str, evidence_texts: Sequence[str]) -> dict:
    """忠实度的**规则基线**：字面重叠启发式，给影子评测当对照。

    规则：把陈述切成 2 字滑窗（近似词），看多大比例出现在证据文本里。
    它完全不懂语义——隐喻、换名、指代都会被低估，数字篡改反而可能因为
    数字本身对得上而漏报。这些系统性偏差正是影子评测要量化的东西，
    所以它只配当基线，永远不该单独决定一条陈述的真假。
    """
    joined = "".join(evidence_texts)
    clean = re.sub(r"\s", "", statement)
    if len(clean) < 2 or not joined:
        return {"label": "uncertain", "reason": "陈述过短或无证据，规则无法判断"}
    grams = [clean[i : i + 2] for i in range(len(clean) - 1)]
    covered = sum(1 for gram in set(grams) if gram in joined)
    ratio = covered / len(set(grams))
    if ratio >= 0.8:
        return {"label": "supported", "reason": f"字面 2 元覆盖 {ratio:.0%}，≥80%"}
    if ratio == 0.0:
        return {"label": "unsupported", "reason": "陈述与证据零字面重叠"}
    return {
        "label": "uncertain",
        "reason": f"字面 2 元覆盖 {ratio:.0%}，介于阈值之间，需人工或 Judge 复核",
    }


# ---------------------------------------------------------------- 完整性


def split_statements(answer: str) -> list[str]:
    """按句号/问号/叹号把回答切成句子列表（保留标点，丢弃空段）。"""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(answer.strip()) if part.strip()]


def evaluate_completeness(
    answer: str, exempt_phrases: Sequence[str] | None = None
) -> dict:
    """完整性指标：事实性陈述里有多大比例缺少任何 ``[n]`` 引用。

    分母是"非豁免陈述"：寒暄/元描述/拒答句按豁免词表跳过（见
    DEFAULT_EXEMPT_PHRASES 的注释）。没有任何待评陈述时 uncited_ratio 为
    None，与"全部缺引用"区分开。
    """
    phrases = tuple(exempt_phrases) if exempt_phrases is not None else DEFAULT_EXEMPT_PHRASES
    statements = split_statements(answer)
    factual = []
    exempted = []
    for statement in statements:
        if any(phrase in statement for phrase in phrases):
            exempted.append(statement)
        else:
            factual.append(statement)
    cited = [s for s in factual if _CITATION_RE.search(s)]
    uncited = [s for s in factual if not _CITATION_RE.search(s)]
    return {
        "statement_count": len(statements),
        "exempted_count": len(exempted),
        "factual_statement_count": len(factual),
        "cited_statement_count": len(cited),
        "uncited_statement_count": len(uncited),
        # None 表示"这条回答没有可评估的事实陈述"（例如纯拒答），不是 100% 缺失
        "uncited_ratio": (len(uncited) / len(factual)) if factual else None,
        "uncited_statements": uncited,
    }


def evaluate_citations(
    answer: str,
    sources: Sequence[Mapping],
    expected_keywords: Sequence[str] = (),
    exempt_phrases: Sequence[str] | None = None,
) -> dict:
    """计算一条回答的引用指标，来源编号按界面约定从 1 开始。

    各返回字段含义：
    - ``valid_number_ratio``：合法编号占全部引用编号的比例（1.0 = 没有越界引用）
      —— 这是三分类里的「正确性」，也是 M3.5 之前的唯一规则指标
    - ``cited_source_count``：实际被引用的不同来源个数
    - ``expected_evidence_coverage``：期望证据关键词出现在**被引用片段**中的比例；
      没有声明期望关键词时为 None，与"覆盖率为 0"区分开
    - ``completeness``：「完整性」分组字典（见 evaluate_completeness）
    - ``manual_support_review_required``：恒为 True。「忠实度」不在本函数计算，
      走 judge_support 影子接口（只在评测脚本里用），见模块 docstring
    """
    # 回答里出现的全部引用编号。dict.fromkeys 去重但保留首次出现顺序——
    # 同一 [1] 引用三次只算一个来源，重复出现不应虚增 cited_source_count。
    mentioned = [int(value) for value in _CITATION_RE.findall(answer)]
    unique_mentioned = list(dict.fromkeys(mentioned))
    # 编号是否落在 [1, len(sources)] 内。界面只把合法编号变成可点击按钮，
    # 越界编号（模型幻觉出的 [99]）在这里单独统计而不是悄悄丢弃。
    valid = [number for number in unique_mentioned if 1 <= number <= len(sources)]
    invalid = [number for number in unique_mentioned if number not in valid]
    cited_sources = [sources[number - 1] for number in valid]

    # 关键词只检查**被引用的**片段，而不是全部来源：如果证据在来源里、
    # 却没出现在被引用的片段中，说明模型找对了材料但没引用对，同样算未覆盖。
    covered_keywords = [
        keyword
        for keyword in expected_keywords
        if any(keyword in str(source.get("text") or source.get("excerpt") or "") for source in cited_sources)
    ]
    keyword_coverage = (
        len(covered_keywords) / len(expected_keywords) if expected_keywords else None
    )

    return {
        # ---- 正确性（原有字段，向后兼容保留）----
        "citation_numbers": unique_mentioned,
        "valid_citation_numbers": valid,
        "invalid_citation_numbers": invalid,
        "valid_number_ratio": (
            len(valid) / len(unique_mentioned) if unique_mentioned else 0.0
        ),
        "cited_source_count": len(valid),
        "expected_keywords": list(expected_keywords),
        "covered_expected_keywords": covered_keywords,
        "expected_evidence_coverage": keyword_coverage,
        # ---- 完整性（M3.5 新增分组）----
        "completeness": evaluate_completeness(answer, exempt_phrases),
        # ---- 忠实度 ----
        # 不在此处计算：需要语义判断，走 judge_support 影子接口（仅评测用）。
        "faithfulness_method": "shadow_only_judge_support",
        "manual_support_review_required": True,
    }
