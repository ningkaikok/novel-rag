"""将 data/novels 下的小说文本切分、向量化并写入本地 Chroma 库。

用法: python src/ingest.py
"""
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, NOVELS_DIR
from loader import load_novel_chunks


def build_index(model: SentenceTransformer | None = None) -> dict:
    """重建向量索引。model 可传入已加载好的 SentenceTransformer 以避免重复加载。

    返回 {"novels": [...], "chunk_count": int}；没有找到任何小说文本时 chunk_count 为 0。
    """
    chunks = load_novel_chunks(NOVELS_DIR)
    if not chunks:
        return {"novels": [], "chunk_count": 0}

    novels = sorted({c.novel for c in chunks})
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    collection.add(
        ids=[f"{c.novel}-{c.chunk_id}" for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"novel": c.novel, "chunk_id": c.chunk_id} for c in chunks],
    )
    return {"novels": novels, "chunk_count": collection.count()}


if __name__ == "__main__":
    result = build_index()
    if result["chunk_count"] == 0:
        print(f"未在 {NOVELS_DIR} 找到任何 .txt 文件，请先放入小说文本。")
    else:
        print(f"完成，来自小说 {result['novels']}，"
              f"已写入 {result['chunk_count']} 条记录到 {CHROMA_DIR}")
