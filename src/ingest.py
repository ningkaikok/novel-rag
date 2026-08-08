"""将 data/novels 下的小说文本切分、向量化并写入 PostgreSQL + pgvector。

同时建两套索引，它们服务于两种互补的检索方式：
- **向量索引**（HNSW）：按语义找，"讲的是同一个意思"就能命中，但对专有名词、
  人名这类"必须逐字匹配"的东西不可靠。
- **BM25 倒排索引**：按词精确匹配并加权，专有名词、人名的强项，但完全不懂近义。

两套索引必须基于同一批文本同时重建，否则检索结果会自相矛盾。

开启 Contextual Retrieval 时（CONTEXTUAL_ENABLED=1），还会给"看不出在讲谁"的
片段生成一句上下文说明，**索引「说明 + 原文」但 text 列仍存原文**——
回答时用原文，不把生成的说明混进正文。

用法: python src/ingest.py
"""
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import (
    CONTEXTUAL_ENABLED,
    CONTEXTUAL_MAX_CHUNKS_PER_BOOK,
    CONTEXTUAL_MODEL,
    CONTEXTUAL_WORKERS,
    NOVELS_DIR,
)
from contextualizer import (
    build_window,
    extract_main_characters,
    generate_contexts_parallel,
    is_context_poor,
    text_hash,
)
from embedder import load_embedder
from loader import load_novel_chunks
from postgres import (
    connect,
    ensure_context_cache,
    load_cached_contexts,
    recreate_schema,
    save_contexts,
    vector_literal,
)
from tokenizer import term_frequencies

# 生成上下文要调 backend 里的模型客户端。src/ 平时不依赖 backend/，
# 这里显式把项目根加进 path，只在真的要用时才导入（见 _make_generate_fn）。
ROOT = Path(__file__).resolve().parent.parent


def _make_generate_fn():
    """返回一个 (prompt) -> 文本流 的函数，用于生成上下文说明。

    延迟导入：只有真的开启 Contextual Retrieval 时才需要 backend 里的模型客户端，
    平时（以及跑测试时）不该因为这个可选功能而引入依赖。
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # 必须先加载 .env 再导入模型客户端。踩过的坑：ingest.py 独立运行时
    # （python src/ingest.py）没有走 FastAPI 的 lifespan，环境里没有
    # ZHIPU_API_KEY，结果 451 次生成**全部失败**——而且因为失败是静默降级的，
    # 日志里只有一句"451 条生成失败"，完全看不出是缺 key。
    from backend.dotenv_lite import load_env

    load_env(ROOT / ".env")
    from backend import claude_cli, zhipu

    if CONTEXTUAL_MODEL.startswith(zhipu.MODEL_PREFIX):
        return lambda prompt: zhipu.generate_stream(prompt, CONTEXTUAL_MODEL)
    if CONTEXTUAL_MODEL.startswith(claude_cli.MODEL_PREFIX):
        return lambda prompt: claude_cli.generate_stream(prompt, CONTEXTUAL_MODEL)
    raise RuntimeError(
        f"CONTEXTUAL_MODEL={CONTEXTUAL_MODEL} 不是支持的模型前缀（需要 glm: 或 claude:）"
    )


def _build_contexts(chunks: list) -> dict[str, str]:
    """给"看不出在讲谁"的片段生成上下文说明，返回 {片段哈希: 说明}。

    三道成本闸门，缺一不可（理由见 contextualizer.py 的模块 docstring）：
    1. **按书跳过**：超过 CONTEXTUAL_MAX_CHUNKS_PER_BOOK 的书直接不做
    2. **选择性**：只处理判定为缺上下文的片段（实测约占 35%）
    3. **增量复用**：已经生成过的（按内容哈希查）直接复用，不重复调 LLM
    """
    ensure_context_cache()

    by_novel: dict[str, list] = {}
    for chunk in chunks:
        by_novel.setdefault(chunk.novel, []).append(chunk)

    # 先挑出需要生成的片段（还没过闸门 3）
    pending: list[tuple[str, str, str]] = []  # (书名, 原文, 窗口)
    pending_hashes: list[str] = []
    for novel, novel_chunks in by_novel.items():
        if len(novel_chunks) > CONTEXTUAL_MAX_CHUNKS_PER_BOOK:
            print(
                f"  跳过《{novel[:16]}》：{len(novel_chunks)} 个片段，"
                f"超过上限 {CONTEXTUAL_MAX_CHUNKS_PER_BOOK}（成本太高）"
            )
            continue
        main_chars = extract_main_characters([c.text for c in novel_chunks])
        poor_indices = [
            i for i, c in enumerate(novel_chunks) if is_context_poor(c.text, main_chars)
        ]
        print(
            f"  《{novel[:16]}》：{len(novel_chunks)} 个片段，"
            f"其中 {len(poor_indices)} 个缺上下文（{len(poor_indices)/len(novel_chunks)*100:.1f}%）"
        )
        for i in poor_indices:
            chunk = novel_chunks[i]
            pending.append((novel, chunk.text, build_window(novel_chunks, i)))
            pending_hashes.append(text_hash(chunk.text))

    if not pending:
        return {}

    # 闸门 3：复用已缓存的结果
    cached = load_cached_contexts(pending_hashes)
    todo = [
        (task, h) for task, h in zip(pending, pending_hashes) if h not in cached
    ]
    print(
        f"  需要生成 {len(pending)} 条，其中 {len(cached)} 条可复用缓存，"
        f"实际调用 LLM {len(todo)} 次"
    )
    if not todo:
        return cached

    generated, errors = generate_contexts_parallel(
        [task for task, _ in todo],
        _make_generate_fn(),
        max_workers=CONTEXTUAL_WORKERS,
    )
    new_items = [(h, ctx) for (_, h), ctx in zip(todo, generated)]
    save_contexts(new_items)

    failed = sum(1 for _, ctx in new_items if not ctx)
    if failed:
        # 失败降级而不是阻断：这些片段照常索引原文，只是没有上下文增强。
        # 但**必须把原因打出来**——只报个数字会让人完全无从排查。
        print(f"  {failed} 条生成失败（已降级为索引原文，下次重建会重试）")
        distinct = sorted(set(errors))
        for reason in distinct[:3]:
            print(f"    失败原因：{reason[:120]}")
        if len(distinct) > 3:
            print(f"    …另有 {len(distinct) - 3} 种其他原因")

    return {**cached, **{h: c for h, c in new_items if c}}


def build_index(model: SentenceTransformer | None = None) -> dict:
    """重建向量索引和 BM25 倒排索引。model 可传入已加载好的 SentenceTransformer 以避免重复加载。

    返回 {"novels": [...], "chunk_count": int}；没有找到任何小说文本时 chunk_count 为 0。
    """
    chunks = load_novel_chunks(NOVELS_DIR)
    if not chunks:
        return {"novels": [], "chunk_count": 0}

    novels = sorted({c.novel for c in chunks})

    contexts: dict[str, str] = {}
    if CONTEXTUAL_ENABLED:
        print("Contextual Retrieval 已开启，正在准备上下文说明…")
        contexts = _build_contexts(chunks)

    # 关键：索引「上下文说明 + 原文」，但下面存进 text 列的仍是原文。
    # 生成的说明只用来改变"这个片段能被什么查询命中"，不该混进送给大模型的正文。
    indexed_texts = [
        f"{contexts[text_hash(c.text)]}\n{c.text}"
        if text_hash(c.text) in contexts
        else c.text
        for c in chunks
    ]

    if model is None:
        model = load_embedder()
    embeddings = model.encode(
        indexed_texts, normalize_embeddings=True, show_progress_bar=True
    )
    dimension = len(embeddings[0])

    # 分词也用加工后的文本，这样上下文说明里的人名同样能被 BM25 命中
    print(f"正在分词 {len(chunks)} 个片段（用于 BM25 索引）…")
    per_chunk_terms = [term_frequencies(t) for t in indexed_texts]
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
    return {"novels": novels, "chunk_count": count, "contextualized": len(contexts)}


if __name__ == "__main__":
    result = build_index()
    if result["chunk_count"] == 0:
        print(f"未在 {NOVELS_DIR} 找到任何小说文本，请先放入小说文本。")
    else:
        extra = (
            f"，其中 {result['contextualized']} 个片段做了上下文增强"
            if result.get("contextualized")
            else ""
        )
        print(
            f"完成，来自小说 {result['novels']}，"
            f"已写入 PostgreSQL novel_chunks 表 {result['chunk_count']} 条记录{extra}"
        )
