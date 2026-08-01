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
# 命中片段前后额外带入的相邻片段数量。问答上下文更完整，但不会把整本书塞给模型。
CONTEXT_NEIGHBORS = int(os.environ.get("CONTEXT_NEIGHBORS", 1))
