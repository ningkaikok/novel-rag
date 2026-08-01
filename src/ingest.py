"""将 data/novels 下的小说文本切分、向量化并写入 PostgreSQL + pgvector。

用法: python src/ingest.py
"""
from sentence_transformers import SentenceTransformer

from config import NOVELS_DIR
from embedder import load_embedder
from loader import load_novel_chunks
from postgres import connect, recreate_schema, vector_literal


def build_index(model: SentenceTransformer | None = None) -> dict:
    """重建向量索引。model 可传入已加载好的 SentenceTransformer 以避免重复加载。

    返回 {"novels": [...], "chunk_count": int}；没有找到任何小说文本时 chunk_count 为 0。
    """
    chunks = load_novel_chunks(NOVELS_DIR)
    if not chunks:
        return {"novels": [], "chunk_count": 0}

    novels = sorted({c.novel for c in chunks})
    if model is None:
        model = load_embedder()

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dimension = len(embeddings[0])
    recreate_schema(dimension)

    rows = [
        (c.novel, c.chunk_id, c.text, vector_literal(embedding))
        for c, embedding in zip(chunks, embeddings)
    ]
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO novel_chunks (novel, chunk_id, text, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                rows,
            )
        count = conn.execute("SELECT count(*) AS count FROM novel_chunks").fetchone()["count"]
    return {"novels": novels, "chunk_count": count}


if __name__ == "__main__":
    result = build_index()
    if result["chunk_count"] == 0:
        print(f"未在 {NOVELS_DIR} 找到任何小说文本，请先放入小说文本。")
    else:
        print(
            f"完成，来自小说 {result['novels']}，"
            f"已写入 PostgreSQL novel_chunks 表 {result['chunk_count']} 条记录"
        )
