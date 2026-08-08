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

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))       # 每个片段的字符数上限
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))  # 相邻片段的重叠字符数

# 本地 Ollama 服务
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

TOP_K = int(os.environ.get("TOP_K", 5))
# 召回候选数量。候选池大于最终上下文，避免过早截断相关片段。
RECALL_K = int(os.environ.get("RECALL_K", 20))
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

# 后端日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# PostgreSQL 连接池大小（只有 FastAPI 后端会用到；独立脚本不建池子）
DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", 1))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", 10))
