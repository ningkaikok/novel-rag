"""Embedding 模型加载：优先用本地缓存，缓存缺失才联网下载。

为什么需要这层封装：
直接 `SentenceTransformer(name)` 每次都会先去 huggingface.co 校验版本。
在网络不通或被墙时，huggingface_hub 会按 1+2+4+8+8 秒退避重试，
**即使模型已经完整缓存在本地**，启动也要白等 20 秒以上；
日志里还会打出误导性的 "Creating a new one with mean pooling"。

因此这里先以 local_files_only 尝试加载，只有本地确实没有时才走联网下载。
"""
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """加载 embedding 模型。已缓存时不访问网络，未缓存时自动下载。"""
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        # 本地没有（或缓存不完整）：联网下载一次，之后就能走上面的快路径
        return SentenceTransformer(model_name)
