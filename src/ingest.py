"""将 data/novels 下的小说文本切分、向量化并写入 PostgreSQL + pgvector。

基础层同时建两套索引，它们服务于两种互补的检索方式：
- **向量索引**（HNSW）：按语义找，"讲的是同一个意思"就能命中，但对专有名词、
  人名这类"必须逐字匹配"的东西不可靠。
- **BM25 倒排索引**：按词精确匹配并加权，专有名词、人名的强项，但完全不懂近义。

两套索引必须基于同一批文本同时切换，否则检索结果会自相矛盾。当前实现按文件哈希
只处理变化书，并用单书 PostgreSQL 事务原子替换向量、BM25、关系边和 manifest。

M3 还会在基础片段之上建立“章节摘要 → 全书摘要”导航层。它有独立流水线指纹：
摘要算法改变时，未变化的书只重算几千个层级节点，不重算几万个基础片段。

开启 Contextual Retrieval 时（CONTEXTUAL_ENABLED=1），还会给"看不出在讲谁"的
片段生成一句上下文说明，**索引「说明 + 原文」但 text 列仍存原文**——
回答时用原文，不把生成的说明混进正文。

建议按下面的数据流阅读 ``build_index``：

    扫描 .txt + manifest
        → 只选 added / modified / deleted
        → 单书切分
        → 可选上下文增强
        → 分批计算 embedding
        → 统计 BM25 词频
        → 可选人物关系边
        → 建立章节/全书摘要并计算 embedding
        → 单书事务原子替换全部派生数据

前面的计算可以很慢，但都在事务外准备，不会长时间锁住旧索引；最后一步才开启短事务。
这是一种适合 RAG/Agent 长任务的通用模式：**先计算，后原子发布**。

用法: python src/ingest.py
"""
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GRAPH_ENABLED,
    GRAPH_MAX_CHUNKS_PER_RELATION,
    GRAPH_MIN_NAME_HITS,
    GRAPH_MODEL,
    HIERARCHY_ENABLED,
    CONTEXTUAL_ENABLED,
    CONTEXTUAL_MAX_CHUNKS_PER_BOOK,
    CONTEXTUAL_MODEL,
    CONTEXTUAL_WORKERS,
    NOVELS_DIR,
)
from graph import (
    RELATION_KEYWORDS,
    build_edges,
    chunks_with_relation,
    extract_characters_from_chunks,
)
from contextualizer import (
    build_window,
    extract_main_characters,
    generate_contexts_parallel,
    is_context_poor,
    text_hash,
)
from embedder import load_embedder
from hierarchy import build_hierarchy_nodes, hierarchy_pipeline_hash
from loader import load_novel_file
from postgres import (
    delete_novel_index,
    ensure_context_cache,
    ensure_graph_cache,
    ensure_index_schema,
    hierarchy_node_count,
    index_chunk_count,
    indexed_novels,
    load_index_manifest,
    load_hierarchy_manifest,
    load_cached_contexts,
    load_cached_graph_characters,
    replace_novel_index,
    replace_novel_hierarchy,
    save_contexts,
    save_graph_characters,
    vector_literal,
)
from tokenizer import term_frequencies

# 生成上下文要调 backend 里的模型客户端。src/ 平时不依赖 backend/，
# 这里显式把项目根加进 path，只在真的要用时才导入（见 _make_generate_fn）。
ROOT = Path(__file__).resolve().parent.parent

ProgressCallback = Callable[[str, int, str], None]
CancelCheck = Callable[[], None]


class IndexCancelled(RuntimeError):
    """后台索引任务收到用户取消信号。"""


@dataclass(frozen=True)
class IndexPlan:
    """一次目录扫描得到的增量计划。"""

    paths: dict[str, Path]
    source_hashes: dict[str, str]
    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged: list[str]
    pipeline_hash: str
    # 基础片段索引没变，但层级摘要缺失/算法升级，需要无损补建的旧书。
    hierarchy_pending: list[str] = field(default_factory=list)
    hierarchy_hash: str = ""


def _file_hash(path: Path) -> str:
    """按原始字节计算 SHA-256；编码或换行变化也应触发重新切分。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_pipeline_hash() -> str:
    """索引算法配置指纹。

    文件没变但切分大小、embedding 模型或上下文/关系图策略变了，也必须重建。
    显式版本号用于分词规则等代码变化：修改这类逻辑时递增版本即可让所有书失效。
    """
    settings = {
        "version": 2,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "contextual_enabled": CONTEXTUAL_ENABLED,
        "contextual_model": CONTEXTUAL_MODEL if CONTEXTUAL_ENABLED else None,
        "graph_enabled": GRAPH_ENABLED,
        "graph_model": GRAPH_MODEL if GRAPH_ENABLED else None,
    }
    encoded = json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_index(
    novels_dir: Path = NOVELS_DIR,
    *,
    force: bool = False,
    manifest: dict[str, dict] | None = None,
    database_novels: set[str] | None = None,
    hierarchy_manifest: dict[str, dict] | None = None,
) -> IndexPlan:
    """比较目录和数据库清单，分类新增、修改、删除和未变化的书。

    ``manifest``/``database_novels`` 参数主要让纯单元测试无需 PostgreSQL；生产路径
    省略它们时读取真实数据库。旧索引没有 manifest 时，已有书会被视为修改并完成
    一次性迁移，而不是错误地假设旧索引一定由当前配置生成。
    """
    external_state = manifest is not None or database_novels is not None
    paths = {path.stem: path for path in sorted(novels_dir.glob("*.txt"))}
    hashes = {novel: _file_hash(path) for novel, path in paths.items()}
    manifest = load_index_manifest() if manifest is None else manifest
    database_novels = indexed_novels() if database_novels is None else database_novels
    pipeline = index_pipeline_hash()

    added: list[str] = []
    modified: list[str] = []
    unchanged: list[str] = []
    for novel in sorted(paths):
        previous = manifest.get(novel)
        if previous is None and novel in database_novels:
            # M2 之前的旧索引没有 manifest：内容还在，但需要做一次迁移以建立清单。
            modified.append(novel)
        elif previous is None or novel not in database_novels:
            added.append(novel)
        elif (
            force
            or previous["source_hash"] != hashes[novel]
            or previous["pipeline_hash"] != pipeline
        ):
            modified.append(novel)
        else:
            unchanged.append(novel)
    deleted = sorted((set(manifest) | database_novels) - set(paths))
    hierarchy_hash = hierarchy_pipeline_hash() if HIERARCHY_ENABLED else ""
    if HIERARCHY_ENABLED:
        if hierarchy_manifest is None:
            # 单元测试显式传入基础清单时，不应再偷偷连接真实 PostgreSQL。
            hierarchy_manifest = {} if external_state else load_hierarchy_manifest()
        hierarchy_pending = [
            novel
            for novel in unchanged
            if not (previous := hierarchy_manifest.get(novel))
            or previous.get("source_hash") != hashes[novel]
            or previous.get("pipeline_hash") != hierarchy_hash
        ]
    else:
        hierarchy_pending = []
    return IndexPlan(
        paths,
        hashes,
        added,
        modified,
        deleted,
        unchanged,
        pipeline,
        hierarchy_pending,
        hierarchy_hash,
    )


def _make_generate_fn(model: str = CONTEXTUAL_MODEL):
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

    if model.startswith(zhipu.MODEL_PREFIX):
        return lambda prompt: zhipu.generate_stream(prompt, model)
    if model.startswith(claude_cli.MODEL_PREFIX):
        return lambda prompt: claude_cli.generate_stream(prompt, model)
    raise RuntimeError(f"{model} 不是支持的模型前缀（需要 glm: 或 claude:）")


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


def _noop_cancel() -> None:
    return None


def _emit(
    callback: ProgressCallback | None, stage: str, percent: int, message: str
) -> None:
    if callback:
        callback(stage, max(0, min(100, int(percent))), message)


def _embedding_dimension(model: SentenceTransformer) -> int:
    dimension = model.get_sentence_embedding_dimension()
    if not dimension:
        raise RuntimeError("无法读取 embedding 模型维度")
    return int(dimension)


def _prepare_hierarchy_rows(
    chunks: list,
    model: SentenceTransformer,
    check: CancelCheck,
) -> list[tuple]:
    """构造章节/全书摘要并分批向量化，返回数据库写入行。"""
    nodes = build_hierarchy_nodes(chunks)
    if not nodes:
        return []
    embeddings: list = []
    batch_size = 32
    for start in range(0, len(nodes), batch_size):
        check()
        batch = nodes[start : start + batch_size]
        embeddings.extend(
            model.encode(
                [node.summary for node in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
    return [
        (
            node.novel,
            node.level,
            node.node_id,
            node.title,
            node.node_order,
            node.start_chunk_id,
            node.end_chunk_id,
            node.summary,
            vector_literal(embedding),
        )
        for node, embedding in zip(nodes, embeddings)
    ]


def build_index(
    model: SentenceTransformer | None = None,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    novels_dir: Path = NOVELS_DIR,
) -> dict:
    """增量同步小说目录和 PostgreSQL 索引。

    变化检测靠文件 SHA-256 + 索引流水线指纹。每本书先在事务外完成切分、Embedding、
    分词和可选关系抽取，最后才用一个事务替换该书的向量/BM25/关系边/manifest。
    因此取消或失败不会清空其他书，也不会留下当前书的半套索引。
    """
    check = cancel_check or _noop_cancel
    _emit(progress, "scan", 1, "正在比较小说文件和索引清单")
    check()
    plan = plan_index(novels_dir, force=force)
    changed = plan.added + plan.modified
    hierarchy_only = [n for n in plan.hierarchy_pending if n not in changed]
    work_total = len(plan.deleted) + len(changed) + len(hierarchy_only)

    # 完全空的新项目不必为了创建空表加载 embedding 模型。
    if work_total == 0:
        count = index_chunk_count()
        hierarchy_count = hierarchy_node_count() if HIERARCHY_ENABLED else 0
        _emit(progress, "complete", 100, "所有小说都已是最新版本")
        return {
            "novels": sorted(plan.paths),
            "chunk_count": count,
            "added": [],
            "modified": [],
            "deleted": [],
            "unchanged": plan.unchanged,
            "contextualized": 0,
            "relations": 0,
            "hierarchy_nodes": hierarchy_count,
        }

    if model is None:
        model = load_embedder()
    ensure_index_schema(_embedding_dimension(model))

    contextualized = 0
    edge_count = 0
    hierarchy_count = 0
    completed_units = 0

    def book_progress(stage: str, local_percent: int, message: str) -> None:
        # 扫描占前 4%，所有书的实际处理共享中间 94%，收尾占最后 2%。
        fraction = (completed_units + local_percent / 100) / max(work_total, 1)
        _emit(progress, stage, 4 + int(fraction * 94), message)

    # 文件已经不存在的书只需要一个短事务，先处理可让书架和检索尽快一致。
    for novel in plan.deleted:
        check()
        book_progress("cleanup", 15, f"正在移除《{novel}》的旧索引")
        delete_novel_index(novel, check)
        book_progress("cleanup", 100, f"已移除《{novel}》")
        completed_units += 1

    for novel in changed:
        check()
        path = plan.paths[novel]
        action = "新增" if novel in plan.added else "更新"
        book_progress("split", 3, f"正在{action}《{novel}》：识别章节并切分")
        chunks = load_novel_file(path)
        check()
        if not chunks:
            raise RuntimeError(f"《{novel}》没有可索引的文本内容")

        contexts: dict[str, str] = {}
        if CONTEXTUAL_ENABLED:
            book_progress("context", 10, f"正在为《{novel}》补充上下文说明")
            contexts = _build_contexts(chunks)
            contextualized += len(contexts)
            check()

        indexed_texts = [
            f"{contexts[text_hash(c.text)]}\n{c.text}"
            if text_hash(c.text) in contexts
            else c.text
            for c in chunks
        ]

        embeddings: list = []
        # 分批不只是为了内存：每批之间都会检查取消信号。若一次把整本书交给
        # model.encode，用户点“停止”后必须等整本推理结束，界面会像失去响应。
        batch_size = 32
        for start in range(0, len(indexed_texts), batch_size):
            check()
            batch = indexed_texts[start : start + batch_size]
            encoded = model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False
            )
            embeddings.extend(encoded)
            done = min(start + len(batch), len(indexed_texts))
            local = 15 + int(done / len(indexed_texts) * 50)
            book_progress(
                "embedding",
                local,
                f"《{novel}》Embedding {done}/{len(indexed_texts)}",
            )

        per_chunk_terms: list[dict[str, int]] = []
        token_counts: list[int] = []
        for index, text in enumerate(indexed_texts, start=1):
            if index % 25 == 0:
                check()
            frequencies = term_frequencies(text)
            per_chunk_terms.append(frequencies)
            token_counts.append(sum(frequencies.values()))
            if index % 100 == 0 or index == len(indexed_texts):
                local = 65 + int(index / len(indexed_texts) * 18)
                book_progress(
                    "bm25",
                    local,
                    f"《{novel}》BM25 分词 {index}/{len(indexed_texts)}",
                )

        relations: list[tuple[str, str, str, str, int]] = []
        if GRAPH_ENABLED:
            check()
            book_progress("graph", 85, f"正在更新《{novel}》的人物关系图")
            relations = _build_relation_edges(chunks)
            edge_count += len(relations)

        hierarchy_rows: list[tuple] | None = None
        if HIERARCHY_ENABLED:
            check()
            book_progress("hierarchy", 86, f"正在建立《{novel}》的章节与全书摘要")
            hierarchy_rows = _prepare_hierarchy_rows(chunks, model, check)
            hierarchy_count += len(hierarchy_rows)

        rows = [
            (
                chunk.novel,
                chunk.chunk_id,
                chunk.chapter_title,
                chunk.text,
                vector_literal(embedding),
                token_count,
                contexts.get(text_hash(chunk.text), ""),
            )
            for chunk, embedding, token_count in zip(
                chunks, embeddings, token_counts
            )
        ]
        check()
        book_progress("database", 92, f"正在原子写入《{novel}》的片段与层级索引")
        replace_novel_index(
            novel,
            rows,
            per_chunk_terms,
            plan.source_hashes[novel],
            plan.pipeline_hash,
            relations,
            check,
            hierarchy_rows,
            plan.hierarchy_hash,
        )
        book_progress("database", 100, f"《{novel}》已安全切换到新索引")
        completed_units += 1

    # M3 之前已经存在且文件未变化的书，只补建层级摘要。基础向量和 BM25 保持原样，
    # 因而升级成本与“章节数”相关，而不是与“片段数”相关。
    for novel in hierarchy_only:
        check()
        book_progress("hierarchy", 10, f"正在补建《{novel}》的章节与全书摘要")
        chunks = load_novel_file(plan.paths[novel])
        rows = _prepare_hierarchy_rows(chunks, model, check)
        book_progress("hierarchy", 85, f"正在写入《{novel}》的 {len(rows)} 个层级节点")
        replace_novel_hierarchy(
            novel,
            rows,
            plan.source_hashes[novel],
            plan.hierarchy_hash,
            check,
        )
        hierarchy_count += len(rows)
        book_progress("hierarchy", 100, f"《{novel}》的层级摘要已可检索")
        completed_units += 1

    check()
    count = index_chunk_count()
    _emit(progress, "complete", 100, f"索引同步完成，共 {count} 个片段")
    return {
        "novels": sorted(plan.paths),
        "chunk_count": count,
        "added": plan.added,
        "modified": plan.modified,
        "deleted": plan.deleted,
        "unchanged": plan.unchanged,
        "contextualized": contextualized,
        "relations": edge_count,
        "hierarchy_nodes": hierarchy_node_count() if HIERARCHY_ENABLED else 0,
    }


def _build_relation_edges(chunks: list) -> list[tuple[str, str, str, str, int]]:
    """按 (书, 关系类型) 抽人物关系边，在原子写入前返回全部边。

    **只从「含关系词的片段」里抽人名**，这是整个设计的关键（详见 graph.py
    的模块 docstring）：关系本来就只存在于这些片段里，从全书均匀采样既贵
    又抽不到关键角色——实测均匀采样 60 段（占全书 0.3%）时，南宫婉这样的
    主要角色根本抽不到。

    成本闸门：每个 (书, 关系) 最多采样 GRAPH_MAX_CHUNKS_PER_RELATION 个片段。
    「师父」这类词能命中上千个片段，不设上限会让建图和全库抽取一样贵。
    """
    ensure_graph_cache()
    generate_fn = None  # 延迟创建：全部命中缓存时根本不需要模型客户端

    by_novel: dict[str, list] = {}
    for chunk in chunks:
        by_novel.setdefault(chunk.novel, []).append(chunk)

    all_edges: list[tuple] = []
    errors: list[str] = []
    reused = 0
    for novel, novel_chunks in by_novel.items():
        for relation in RELATION_KEYWORDS:
            matched = chunks_with_relation(novel_chunks, relation)
            if len(matched) < 2:
                continue
            # 超过上限时均匀抽样，保证覆盖全书而不是只取开头
            if len(matched) > GRAPH_MAX_CHUNKS_PER_RELATION:
                step = len(matched) // GRAPH_MAX_CHUNKS_PER_RELATION
                matched = matched[::step][:GRAPH_MAX_CHUNKS_PER_RELATION]

            # 增量：按「采样到的片段内容」做哈希查缓存。
            # 加新书时老书的哈希不变、直接复用；改了切分参数则哈希变化、
            # 会重抽——那是对的，内容确实变了。
            sample_hash = text_hash("\n".join(c.text for c in matched))
            cached = load_cached_graph_characters(novel, relation, sample_hash)
            if cached is not None:
                characters = set(cached)
                reused += 1
            else:
                if generate_fn is None:
                    generate_fn = _make_generate_fn(GRAPH_MODEL)
                name_hits = extract_characters_from_chunks(
                    matched, generate_fn, errors=errors
                )
                characters = {
                    n for n, hits in name_hits.items() if hits >= GRAPH_MIN_NAME_HITS
                }
                save_graph_characters(novel, relation, sample_hash, sorted(characters))
            if len(characters) < 2:
                continue
            edges = build_edges(matched, characters, relation)
            all_edges.extend(edges)
            print(
                f"  《{novel[:12]}》{relation}：{len(matched)} 段 → "
                f"{len(characters)} 个人物 → {len(edges)} 条边"
            )

    if reused:
        print(f"  {reused} 组「书×关系」直接复用了缓存，没有重复调用 LLM")
    if errors:
        # 降级不能吞掉原因（Contextual Retrieval 那边踩过这个坑）
        distinct = sorted(set(errors))
        print(f"  {len(errors)} 批抽取失败（已跳过，不影响其余部分）")
        for reason in distinct[:2]:
            print(f"    失败原因：{reason[:110]}")

    return all_edges


if __name__ == "__main__":
    last_progress = {"stage": "", "percent": -5}

    def print_progress(stage: str, percent: int, message: str) -> None:
        if (
            stage != last_progress["stage"]
            or percent >= last_progress["percent"] + 5
            or percent == 100
        ):
            print(f"[{percent:3d}%] {message}")
            last_progress.update(stage=stage, percent=percent)

    result = build_index(progress=print_progress)
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
            f"PostgreSQL 当前共 {result['chunk_count']} 条记录{extra}；"
            f"新增 {len(result['added'])}、更新 {len(result['modified'])}、"
            f"删除 {len(result['deleted'])}、未变化 {len(result['unchanged'])}"
        )
