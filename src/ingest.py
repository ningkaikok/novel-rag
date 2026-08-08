"""将 data/novels 下的小说文本切分、向量化并写入 PostgreSQL + pgvector。

同时建两套索引，它们服务于两种互补的检索方式：
- **向量索引**（HNSW）：按语义找，"讲的是同一个意思"就能命中，但对专有名词、
  人名这类"必须逐字匹配"的东西不可靠。
- **BM25 倒排索引**：按词精确匹配并加权，专有名词、人名的强项，但完全不懂近义。

两套索引必须基于同一批文本同时重建，否则检索结果会自相矛盾。

用法: python src/ingest.py
"""
from sentence_transformers import SentenceTransformer

from config import NOVELS_DIR
from embedder import load_embedder
from loader import load_novel_chunks
from postgres import connect, recreate_schema, vector_literal
from tokenizer import term_frequencies


def build_index(model: SentenceTransformer | None = None) -> dict:
    """重建向量索引和 BM25 倒排索引。model 可传入已加载好的 SentenceTransformer 以避免重复加载。

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

    # 先分好词：既要拿到每个片段的词频（写倒排索引），也要拿到词数（BM25 长度归一化）
    print(f"正在分词 {len(chunks)} 个片段（用于 BM25 索引）…")
    per_chunk_terms = [term_frequencies(c.text) for c in chunks]
    token_counts = [sum(tf.values()) for tf in per_chunk_terms]

    recreate_schema(dimension)

    rows = [
        (c.novel, c.chunk_id, c.text, vector_literal(embedding), token_count)
        for c, embedding, token_count in zip(chunks, embeddings, token_counts)
    ]
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO novel_chunks (novel, chunk_id, text, embedding, token_count) "
                "VALUES (%s, %s, %s, %s::vector, %s)",
                rows,
            )

        # 倒排索引约 390 万行。用 COPY 而不是 executemany：后者要为每一行走一次
        # 完整的「发送 SQL → 解析 → 执行」往返，几百万行会慢到不可接受；
        # COPY 是 PostgreSQL 的批量导入协议，数据以流的方式一次灌进去。
        total_terms = sum(len(tf) for tf in per_chunk_terms)
        print(f"正在写入 BM25 倒排索引（约 {total_terms:,} 行）…")
        with conn.cursor() as cursor:
            with cursor.copy(
                "COPY chunk_terms (novel, chunk_id, term, tf) FROM STDIN"
            ) as copy:
                for chunk, freqs in zip(chunks, per_chunk_terms):
                    for term, tf in freqs.items():
                        copy.write_row((chunk.novel, chunk.chunk_id, term, tf))

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
