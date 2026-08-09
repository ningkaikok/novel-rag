import getpass
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NOVELS_DIR = ROOT_DIR / "data" / "novels"

# PostgreSQL + pgvector。可用 DATABASE_URL 覆盖默认本机连接。
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{getpass.getuser()}@127.0.0.1:5432/novel_rag",
)

# 本地 embedding 模型（中文效果较好，体积小）。这是「双编码器」，负责快速粗筛。
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# 本地重排模型（「交叉编码器」，负责对少量候选做精细排序，见 src/reranker.py）。
# 比 embedding 模型大得多（约 1.1GB），但只对 RECALL_K 个候选跑，耗时可接受。
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
# 设成 0 可以完全关掉重排（模型下载不了、或想对比重排前后效果时用）
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") != "0"
# 送进重排的候选数 = 最终要的条数 × 这个倍数。重排要有东西可挑，候选池必须
# 明显大于最终结果——业界经验是「召回 20 → 重排到 5」这个量级，即 3~4 倍。
# 倍数太小重排没得挑，太大则交叉编码器要算的对数线性增加、变慢。
RERANK_CANDIDATE_MULTIPLIER = int(os.environ.get("RERANK_CANDIDATE_MULTIPLIER", 3))

# --- Contextual Retrieval（给缺上下文的片段补一句说明，见 src/contextualizer.py）---
# **默认关闭**：它要调 LLM，实测单条约 4.4 秒。开之前请先看清楚下面两个闸门。
CONTEXTUAL_ENABLED = os.environ.get("CONTEXTUAL_ENABLED", "0") == "1"
# 成本闸门：超过这个片段数的书直接跳过，不做上下文增强。
# 《凡人修仙传》19501 个片段、《诡秘之主》11948 个——就算只处理 35% 也要好几小时，
# 默认值刻意设在它们之下、《降龙》(1278) 之上，避免手一滑跑一整夜。
CONTEXTUAL_MAX_CHUNKS_PER_BOOK = int(
    os.environ.get("CONTEXTUAL_MAX_CHUNKS_PER_BOOK", 2000)
)
# 生成上下文用的模型。用便宜的小模型就够——它只是写一句话，不需要推理能力。
CONTEXTUAL_MODEL = os.environ.get("CONTEXTUAL_MODEL", "glm:glm-4-flash")
# 并发数。实测单次 4.4 秒，451 个片段串行 33 分钟、8 路并发约 4 分钟。
CONTEXTUAL_WORKERS = int(os.environ.get("CONTEXTUAL_WORKERS", 8))

# --- 多轮对话查询改写（把带指代的追问补全，见 src/query_rewriter.py）---
# 默认开启：它只在「有历史 且 问题像是依赖上文」时才触发，第一轮和自足的问题
# 完全不花钱；而不开的话，"他后来怎么样了"这类追问必然检索失败。
QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "1") != "0"
# 改写用的模型。这只是个句子改写任务，用便宜快速的小模型即可——
# 不要用当前对话选的模型：用户可能选了推理型的大模型，改写会白等好几秒。
QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", "glm:glm-4-flash")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))       # 每个片段的字符数上限
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))  # 相邻片段的重叠字符数

# 本地 Ollama 服务
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# 最终送进 prompt 的片段数——这是「长上下文取舍」这个杠杆的主旋钮。
#
# 默认值 3 是**实测出来的**，不是拍脑袋（用 scripts/eval_context_budget.py，
# 13 条用例，开启重排）：
#
#     TOP_K    命中率    prompt字数    信噪比
#       3      0.846      4065       0.356    ← 默认值
#       5      0.846      6665       0.281
#      10      0.846     12464       0.234
#
# 三档命中率完全相同，但 3 比 5 省 39% 的字数，信噪比还更高。
#
# **这正是重排的价值所在**：关掉重排时 3→5→10 的命中率还在涨
# （0.692→0.769→0.846），说明必须多送才能保证答案在里面；开了重排之后，
# 最相关的几段被顶到最前面，送 3 段就够了。
#
# 送更多不是无害的：① 花钱变多、变慢；② 上下文过长时模型对中间部分的注意力
# 会下降（"迷失在中间"）；③ 无关片段稀释信号，增加被带偏和产生幻觉的概率。
TOP_K = int(os.environ.get("TOP_K", 3))
# 召回候选数量。候选池大于最终上下文，避免过早截断相关片段。
RECALL_K = int(os.environ.get("RECALL_K", 20))

# 「长上下文取舍」的另一面：整本书小到能全塞进模型窗口时，**RAG 本身就是多余的**。
# 与其检索出几段（可能漏掉关键信息），不如把全文给模型——不会有任何信息损失。
#
# 阈值按字数算（中文约 0.7 token/字）。8000 字 ≈ 5600 token，远低于主流模型的
# 窗口（GLM-4-Flash 是 12.8 万 token），留足余量给对话历史和输出。
#
# 现状说明：当前语料里只有《雾隐山庄》（1229 字）会触发，另外三本都在 60 万字
# 以上。这个短路主要服务于「用户上传一份小文档」的场景——一篇论文、一份报告
# 通常就在这个量级。
FULL_TEXT_MAX_CHARS = int(os.environ.get("FULL_TEXT_MAX_CHARS", 8000))
# --- BM25 关键词检索的两个调参旋钮 ---
# 这两个值是 BM25 论文里的经典默认值，绝大多数场景不需要动。
#
# k1：词频饱和速度。一个词在片段里出现 10 次，并不意味着相关性是出现 1 次的
# 10 倍——边际收益递减。k1 越小饱和越快（第 2 次出现带来的增益就已经很小），
# 越大越接近线性。取 1.2 是通用经验值。
BM25_K1 = float(os.environ.get("BM25_K1", 1.2))
# b：文档长度归一化强度，取值 0~1。长片段天然更容易碰巧包含查询词，不做归一化
# 的话长片段会系统性占便宜。b=0 完全不归一化，b=1 完全按长度比例惩罚，
# 0.75 是折中的通用默认值。
BM25_B = float(os.environ.get("BM25_B", 0.75))
# 命中片段前后额外带入的相邻片段数量。问答上下文更完整，但不会把整本书塞给模型。
CONTEXT_NEIGHBORS = int(os.environ.get("CONTEXT_NEIGHBORS", 1))

# --- 层级检索（片段 → 章节摘要 → 全书摘要，见 src/hierarchy.py）---
# 默认开启且不调用 LLM：摘要采用可重复、无额外费用的抽取式策略。它们只负责
# 帮系统定位“应该看哪些章节”，最终回答仍使用并引用 novel_chunks 里的原文。
HIERARCHY_ENABLED = os.environ.get("HIERARCHY_ENABLED", "1") != "0"
# 单个摘要控制在 embedding 模型能有效编码的范围内；不是生成模型的上下文上限。
HIERARCHY_SUMMARY_MAX_CHARS = int(
    os.environ.get("HIERARCHY_SUMMARY_MAX_CHARS", 800)
)
# 没有章节标题的 txt 按固定片段窗口构造“虚拟章节”，保证任何小说都能建立层级。
HIERARCHY_UNTITLED_CHUNKS = int(
    os.environ.get("HIERARCHY_UNTITLED_CHUNKS", 12)
)
# 全局问题最多选择多少个章节节点进入候选池。之后还会映射回原文并经过重排。
HIERARCHY_TOP_K = int(os.environ.get("HIERARCHY_TOP_K", 6))

# 后端日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# PostgreSQL 连接池大小（只有 FastAPI 后端会用到；独立脚本不建池子）
DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", 1))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", 10))

# --- GraphRAG 人物关系图（见 src/graph.py）---
# 默认关闭：建图要调 LLM 抽人名。但成本远低于 Contextual Retrieval——
# 只从「含关系词的片段」里抽，实测《凡人修仙传》的「伴侣」关系只要 11 次调用。
GRAPH_ENABLED = os.environ.get("GRAPH_ENABLED", "0") == "1"
# 每个 (书, 关系类型) 最多采样多少个片段去抽人名。成本闸门：
# 「师父」这类词能命中上千个片段，不设上限会让建图变得和全库抽取一样贵。
GRAPH_MAX_CHUNKS_PER_RELATION = int(
    os.environ.get("GRAPH_MAX_CHUNKS_PER_RELATION", 80)
)
# 抽人名用的模型（和 Contextual Retrieval 一样，便宜的小模型就够）
GRAPH_MODEL = os.environ.get("GRAPH_MODEL", "glm:glm-4-flash")
# 一个人名要在几个批次里都被认作人名，才算数。降噪用：
# 单次出现的往往是模型偶然把泛称当成了名字。
GRAPH_MIN_NAME_HITS = int(os.environ.get("GRAPH_MIN_NAME_HITS", 2))
