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

# 本地 embedding 模型（中文效果较好，体积小）
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))       # 每个片段的字符数上限
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))  # 相邻片段的重叠字符数

# 本地 Ollama 服务
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

TOP_K = int(os.environ.get("TOP_K", 5))
# 召回候选数量。候选池大于最终上下文，避免过早截断相关片段。
RECALL_K = int(os.environ.get("RECALL_K", 20))
# 关键词召回按问题分词后逐词匹配；某个词命中的片段数超过这个值就跳过它——
# 太常见的词（比如主角名，几乎每页都出现）起不到筛选作用，反而会把结果
# 变成"这本书随便哪几段"，不如不用它做关键词。
KEYWORD_GENERIC_LIMIT = int(os.environ.get("KEYWORD_GENERIC_LIMIT", 300))
# 问题分词后最多取几个词去查——每个候选词都要先查一次命中数（判断是否太常见），
# 问题很长、分词很碎时词数可能到十几个，全部都查会让一次问答多花好几秒。
# 封顶后优先保留更长的词（通常是人名、专有名词，比短的虚词/动词更有筛选价值）。
KEYWORD_MAX_TERMS = int(os.environ.get("KEYWORD_MAX_TERMS", 6))
# 命中片段前后额外带入的相邻片段数量。问答上下文更完整，但不会把整本书塞给模型。
CONTEXT_NEIGHBORS = int(os.environ.get("CONTEXT_NEIGHBORS", 1))

# 后端日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# PostgreSQL 连接池大小（只有 FastAPI 后端会用到；独立脚本不建池子）
DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", 1))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", 10))
