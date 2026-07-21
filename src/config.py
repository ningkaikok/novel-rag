import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NOVELS_DIR = ROOT_DIR / "data" / "novels"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "novels"

# 本地 embedding 模型（中文效果较好，体积小）
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))       # 每个片段的字符数上限
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))  # 相邻片段的重叠字符数

# 本地 Ollama 服务
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

TOP_K = int(os.environ.get("TOP_K", 5))
