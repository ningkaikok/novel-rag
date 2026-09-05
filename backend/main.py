"""FastAPI 后端：把 src 下的 RAG 逻辑包成 HTTP 接口。

一次提问在这里经历什么（读这个文件建议从 /api/ask 开始）
--------------------------------------------------------
    前端 POST /api/ask
        ↓
    ① choose_answer_route()     自动 / 仅原文 / 自由问答
        ├─ 自由问答 ────────────→ 跳过索引和检索，直接生成
        └─ 原文问答
             ↓
    ② _rewrite_for_search()     多轮追问补全指代（"他"→"李化元"）
        ↓                        ⚠️ 只在原文路径触发，且必须在检索之前
    ③ rag.retrieve_hybrid_stream()   每完成一步就推一条 step 事件
        ├─ 全局问题：全书/章节摘要导航后回到原文
        ├─ 语义检索（向量，pgvector HNSW）
        ├─ BM25 检索（倒排索引）
        ├─ 结构性检索（按 chunk_id 定位开头/结尾）
        ├─ RRF 融合成候选池
        └─ 交叉编码器重排，精选出 top_k
        ↓
    ④ rag.build_answer_context()  上下文组装：默认补相邻片段（off/neighbors 档），
                                  chapter 实验档改为整章扩展 + token 预算闸门
        ↓
    ⑤ 按模型前缀路由到生成后端
        claude: → 本地 Claude CLI ／ glm: → 智谱 ／ 其余 → 本地 Ollama
        ↓
    ⑥ SSE 流式推送
        event: step    → 「思考过程」的一步，也可带候选排名供评测面板复盘
        event: sources → 原文出处卡片
        event: token   → 逐字打字机
        event: done
        ↓
    ⑦ 落库到 chat_turns（带 session_id 时），刷新页面能恢复

为什么 src/ 和 backend/ 分开
-----------------------------
`src/` 是**纯业务逻辑**，不依赖 Web 框架——所以 `python src/ingest.py` 能独立
跑、pytest 也能直接测。`backend/` 只做"把它包成 HTTP"这一件事：路由、流式、
错误信封、会话持久化。这条边界让检索逻辑可以脱离 HTTP 单独验证
（`scripts/eval_retrieval.py` 就是直接调 `src/rag.py`，完全不经过这个文件）。

为什么没有 LangGraph
---------------------
标准 RAG 虽有多个阶段，但方向固定、一次请求内就能结束。独立的 `/api/agent/ask`
确实会“选择工具 → 观察 → 再判断”，但它只有五个只读工具、最多五步、无需检查点，
普通 Python 循环更容易学习和测试。等到需要人工审批、并行子任务或跨进程断点恢复，
再把已测试的工具提升为图节点。决策记录见 `docs/architecture-decisions.md`。

运行：uvicorn backend.main:app --reload --port 8000
（在项目根目录 novel-rag/ 下运行）
"""

import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

# 让后端能 import src 下的业务逻辑（完全复用，不改动）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 先加载 .env，再导入下面读环境变量的模块（config 在导入时就会读取），
# 这样密钥写在 .env 里即可，不必每次启动手动 export。
from backend.dotenv_lite import load_env  # noqa: E402

_ENV_KEYS = load_env(ROOT / ".env")

# 尽早配置好 logging，确保下面（以及 lifespan 里）任何一条日志都带上格式和级别
from backend.logging_config import configure_logging, get_logger  # noqa: E402

configure_logging()
logger = get_logger("main")

import ingest  # noqa: E402
from agent_lab import run_agent  # noqa: E402
from backend import claude_cli, zhipu  # noqa: E402
from backend.errors import APIError, ErrorCode, register_exception_handlers  # noqa: E402
from backend.index_tasks import (  # noqa: E402
    IndexTaskManager,
    PostgresTaskStore,
    TaskAlreadyRunning,
    TaskNotFound,
)
from backend.middleware import RequestIDMiddleware  # noqa: E402
from backend.schemas import (  # noqa: E402
    AgentAskRequest,
    AgentStep,
    AskRequest,
    BookList,
    CurrentModel,
    DeleteResult,
    GraphEdgeItem,
    GraphEdgeList,
    GraphReviewRequest,
    GraphReviewResult,
    HealthStatus,
    IndexTaskStatus,
    ModelList,
    SearchMatch,
    SearchResult,
    SessionHistory,
    SetModelRequest,
    SourceItem,
    StoredTurn,
    TraceStep,
    UploadResult,
    VerifyCitationRequest,
    VerifyCitationResult,
)
from citation_eval import judge_support, statements_citing  # noqa: E402
from config import (  # noqa: E402
    HISTORY_IN_PROMPT,
    MAX_UPLOAD_BYTES,
    NOVELS_DIR,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    QUERY_EXPAND_ENABLED,
    QUERY_EXPAND_MODEL,
    QUERY_REWRITE_ENABLED,
    QUERY_REWRITE_MODEL,
    RERANK_ENABLED,
    RERANKER_MODEL,
)
from embedder import load_embedder  # noqa: E402
from generation_mixin import build_history_block  # noqa: E402
from postgres import (  # noqa: E402
    VALID_REVIEW_STATUSES,
    close_pool,
    connect,
    ensure_chat_schema,
    ensure_graph_review_schema,
    ensure_index_task_schema,
    ensure_novel_metadata_schema,
    has_index,
    init_pool,
    list_relation_edges,
    load_turns,
    next_turn_index,
    save_turn,
    set_relation_review,
)
from query_rewriter import rewrite_query  # noqa: E402
from query_router import AnswerMode, build_free_prompt, choose_answer_route  # noqa: E402
from rag import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    NovelRAG,
    generate_ollama_prompt_stream,
)

# 进程级共享资源（对应 Streamlit 的 cache_resource）
state: dict = {}
index_tasks = IndexTaskManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 只打变量名不打值：既能确认密钥已加载，又不会把密钥写进日志
    if _ENV_KEYS:
        logger.info(f"已从 .env 加载：{', '.join(_ENV_KEYS)}")
    logger.info(f"云端可选模型：{claude_cli.claude_model_options() + zhipu.model_options()}")
    # 连接池要在其他任何用到 connect() 的操作之前建好，这样 ensure_chat_schema、
    # 后面每次请求的检索/会话持久化都能直接复用池子里的连接，不用逐次握手。
    try:
        init_pool()
        logger.info("PostgreSQL 连接池已就绪")
    except Exception as exc:
        # 数据库暂时连不上：不阻断启动，connect() 会退化为逐次新建连接
        # （行为和引入连接池之前完全一样），书架管理等不依赖索引的功能仍可用。
        logger.warning(f"PostgreSQL 连接池初始化失败（退化为逐次新建连接）：{exc}")
    # 对话历史表（幂等建表，和向量索引分开，重建索引不会清空聊天记录）
    try:
        ensure_chat_schema()
    except Exception as exc:
        # 建表失败只影响"刷新后恢复历史"，问答本身不受影响，所以不阻断启动
        logger.warning(f"对话历史表初始化失败（会话持久化不可用）：{exc}")
    # 旧索引没有 chapter_title。先幂等补列保证新查询兼容；值仍为空，用户重建
    # 索引后才会真正得到章节归属。
    try:
        ensure_novel_metadata_schema()
    except Exception as exc:
        logger.warning(f"小说索引元数据升级失败（章节名暂不可用）：{exc}")
    # 索引任务卡片状态落库（M3.3.5）：重启后侧栏能恢复上一次任务卡片；上次进程
    # 遗留的 active 任务会被如实标记为 failed（事务已随连接断开回滚，可安全重试）。
    try:
        ensure_index_task_schema()
        interrupted = index_tasks.set_store(PostgresTaskStore())
        if interrupted:
            logger.info(f"已把 {interrupted} 个上次遗留的索引任务标记为中断，可重试")
    except Exception as exc:
        # 落库不可用只影响"重启恢复卡片"，任务本身照常运行（内存状态足够）
        logger.warning(f"索引任务状态落库初始化失败（重启后不恢复卡片）：{exc}")
    # 人物关系图 schema v2（M4）：审核界面依赖的两张表在这里幂等升级，
    # 让用户不必先重建一次索引就能打开关系审核面板。
    try:
        ensure_graph_review_schema()
    except Exception as exc:
        logger.warning(f"人物关系图 schema 升级失败（关系审核不可用）：{exc}")
    # 启动时加载一次 embedding 模型，并尝试连接 PostgreSQL 索引
    state["embedder"] = load_embedder()
    state["rag"] = _try_load_rag()
    state["model"] = OLLAMA_MODEL  # 当前用于生成回答的模型，可通过 /api/model 动态切换
    yield
    index_tasks.shutdown()
    state.clear()
    close_pool()


def _try_load_rag() -> NovelRAG | None:
    try:
        service = NovelRAG(embedder=state["embedder"])
    except Exception:
        return None  # PostgreSQL 索引还没建立
    # 自适应查询扩展（M3.4）：rag.py 不依赖云端 SDK，生成函数由 Web 层按
    # QUERY_EXPAND_MODEL 前缀路由注入（和上面 _rewrite_for_search 的路由
    # 是同一个模式）。开关默认关闭，关闭时这里什么都不挂、零开销。
    if QUERY_EXPAND_ENABLED:
        service.expand_generate_fn = lambda prompt: (
            zhipu.generate_stream(prompt, QUERY_EXPAND_MODEL)
            if QUERY_EXPAND_MODEL.startswith(zhipu.MODEL_PREFIX)
            else claude_cli.generate_stream(prompt, QUERY_EXPAND_MODEL)
        )
    return service


app = FastAPI(title="书虫 · Novel RAG API", lifespan=lifespan)
register_exception_handlers(app)

# 开发期允许 Vite dev server 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:45173", "http://127.0.0.1:45173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 最后加的中间件在最外层（Starlette 的顺序），让 request-id 覆盖整个请求生命周期
# （包括 CORS 预检），且不是 BaseHTTPMiddleware——不会缓冲 /api/ask 的 SSE 响应体。
app.add_middleware(RequestIDMiddleware)


# ----------------------------------------------------------------- 书架
@app.get("/api/books", response_model=BookList)
def list_books():
    return BookList(books=sorted(p.stem for p in NOVELS_DIR.glob("*.txt")))


async def _read_limited(f: UploadFile, name: str) -> bytes:
    """按 1MB 一块流式读取上传文件，超过 MAX_UPLOAD_BYTES 立即中断。

    不用 `await f.read()` 一次读入：那会把整个文件先放进内存，用户误选一个
    几百 MB 的文件就能打爆进程。分块读让超限在最早的一刻被发现，此时最多
    只浪费了上限+1MB 的内存。
    """
    buf = bytearray()
    while chunk := await f.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise APIError(
                413,
                ErrorCode.file_too_large,
                f"《{name}》超过单文件 {limit_mb}MB 的大小上限，请拆分或转换后再上传",
            )
    return bytes(buf)


@app.post("/api/books", response_model=UploadResult)
async def upload_books(files: list[UploadFile]):
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        name = Path(f.filename or "").name  # 防止路径穿越
        if not name.lower().endswith(".txt"):
            continue
        payloads.append((name, await _read_limited(f, name)))
    if not payloads:
        raise APIError(400, ErrorCode.no_valid_files, "没有有效的 .txt 文件")

    def save_files() -> None:
        """先全部写到同目录临时文件，再用原子 rename 替换正式文件。"""
        NOVELS_DIR.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        try:
            for name, content in payloads:
                target = NOVELS_DIR / name
                temporary = NOVELS_DIR / f".{name}.{uuid.uuid4().hex}.upload"
                temporary.write_bytes(content)
                staged.append((temporary, target))
            for temporary, target in staged:
                temporary.replace(target)
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)

    task = _start_index_task(prepare=save_files)
    return {"saved": [Path(name).stem for name, _ in payloads], "task": task}


@app.delete("/api/books/{name}", response_model=DeleteResult)
def delete_book(name: str):
    # 只允许删除 novels 目录下的 txt，拒绝路径穿越
    target = (NOVELS_DIR / f"{Path(name).name}.txt").resolve()
    if target.parent != NOVELS_DIR.resolve() or not target.exists():
        raise APIError(404, ErrorCode.book_not_found, "书不存在")
    task = _start_index_task(prepare=target.unlink)
    return {"deleted": name, "task": task}


@app.post("/api/reindex", response_model=IndexTaskStatus)
def reindex(force: bool = Query(default=False)):
    """启动后台同步。默认只处理变化文件；force=true 强制重新处理所有现存书。"""
    return _start_index_task(force=force)


def _start_index_task(
    *,
    force: bool = False,
    retry_of: str | None = None,
    prepare=None,
) -> dict:
    """把增量索引包装成任务，并统一映射并发冲突错误。"""

    def build(progress, cancel_check):
        result = ingest.build_index(
            model=state.get("embedder"),
            force=force,
            progress=progress,
            cancel_check=cancel_check,
            novels_dir=NOVELS_DIR,
        )
        # NovelRAG 自身不缓存片段，但首次建库前 state["rag"] 是 None；成功后要补上。
        state["rag"] = _try_load_rag() if result["chunk_count"] else None
        return result

    try:
        return index_tasks.start(build, force=force, retry_of=retry_of, prepare=prepare)
    except TaskAlreadyRunning as exc:
        raise APIError(
            409,
            ErrorCode.index_task_running,
            f"已有索引任务正在运行（{exc.task['progress']}%：{exc.task['message']}）",
        ) from exc


def _get_index_task(task_id: str) -> dict:
    try:
        return index_tasks.get(task_id)
    except TaskNotFound as exc:
        raise APIError(404, ErrorCode.index_task_not_found, "索引任务不存在") from exc


@app.get("/api/index-tasks/current", response_model=IndexTaskStatus | None)
def current_index_task():
    return index_tasks.current()


@app.get("/api/index-tasks/{task_id}", response_model=IndexTaskStatus)
def get_index_task(task_id: str):
    return _get_index_task(task_id)


@app.post("/api/index-tasks/{task_id}/cancel", response_model=IndexTaskStatus)
def cancel_index_task(task_id: str):
    _get_index_task(task_id)
    return index_tasks.cancel(task_id)


@app.post("/api/index-tasks/{task_id}/retry", response_model=IndexTaskStatus)
def retry_index_task(task_id: str):
    previous = _get_index_task(task_id)
    if previous["status"] not in {"failed", "cancelled"}:
        raise APIError(
            409,
            ErrorCode.index_task_not_retryable,
            "只有失败或已取消的索引任务可以重试",
        )
    return _start_index_task(force=previous["force"], retry_of=task_id)


# ----------------------------------------------------------------- 全文搜索
@app.get("/api/search", response_model=SearchResult)
def search(
    q: str = Query(min_length=1, max_length=200),
    book: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """在本地小说原文中做精确全文搜索，不经过大模型。"""
    needle = q.strip().casefold()
    if not needle:
        return SearchResult(query=q, total=0, results=[])

    if not has_index():
        raise APIError(
            409, ErrorCode.index_not_ready, "PostgreSQL 索引未建立，请先重新整理书架"
        )

    if book:
        where_sql = "novel = %s AND position(lower(%s) in lower(text)) > 0"
        query_params = (book, needle)
    else:
        where_sql = "position(lower(%s) in lower(text)) > 0"
        query_params = (needle,)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT novel, chunk_id, chapter_title, text
            FROM novel_chunks
            WHERE {where_sql}
            ORDER BY novel, chunk_id
            """,
            query_params,
        ).fetchall()

    matches = [
        SearchMatch(
            novel=row["novel"],
            chunk_id=int(row["chunk_id"]),
            chapter_title=row.get("chapter_title"),
            text=row["text"],
            match_count=row["text"].casefold().count(needle),
        )
        for row in rows
    ]

    return SearchResult(
        query=q,
        total=len(matches),
        results=matches[offset : offset + limit],
    )


# ----------------------------------------------------------------- 提问（SSE 流式）
def _load_history(session_id: str | None) -> list[dict]:
    """读这个会话的历史；没有 session_id 或读失败都返回空列表。

    M3.6 起历史有两个消费方——查询改写和最终回答的 prompt——所以集中读一次
    再分发，避免同一次请求打两遍数据库。读失败不能影响回答：最坏退回到
    "没有历史"，也就是改造之前的行为，但要留下日志，别静默失效。
    """
    if not session_id:
        return []
    try:
        return load_turns(session_id)
    except Exception as exc:
        logger.warning(f"读会话历史失败（不影响回答）：{exc}")
        return []


def _rewrite_for_search(req: AskRequest, turns: list[dict]) -> str:
    """带上会话历史，把追问补全成能独立检索的问题；任何一步失败都退回原问题。

    整条链路上的每一步都可能失败（没配 session_id、读历史失败、模型限流），
    但**没有一步应该让提问功能不可用**——最坏情况就是退回改造前的行为：
    拿原问题去检索。所以这里层层兜底，但每次兜底都记一条日志，
    避免出现"功能静默失效却查不出原因"（Contextual Retrieval 那边踩过这个坑）。
    """
    if not (QUERY_REWRITE_ENABLED and turns):
        return req.question

    errors: list[str] = []
    rewritten = rewrite_query(
        req.question,
        turns,
        lambda prompt: (
            zhipu.generate_stream(prompt, QUERY_REWRITE_MODEL)
            if QUERY_REWRITE_MODEL.startswith(zhipu.MODEL_PREFIX)
            else claude_cli.generate_stream(prompt, QUERY_REWRITE_MODEL)
        ),
        errors,
    )
    for reason in errors:
        logger.warning(f"查询改写失败，按原问题检索：{reason}")
    if rewritten != req.question:
        logger.info(f"查询改写：「{req.question}」→「{rewritten}」")
    return rewritten


_SENTINEL = object()  # 线程池里取不到下一个 token 时的哨兵，区别于"取到了 None"


def _next_or_sentinel(iterator):
    """在线程里取生成器的下一个元素；取完返回哨兵而不是抛 StopIteration。

    StopIteration 不能穿过 await 边界（会变成 RuntimeError），所以用哨兵传递结束。
    """
    return next(iterator, _SENTINEL)


def _generate_for_model(prompt: str, model: str):
    """统一三种生成后端，普通问答和 Agent Lab 共用同一条模型路由。"""
    if model.startswith(claude_cli.MODEL_PREFIX):
        return claude_cli.generate_stream(prompt, model)
    if model.startswith(zhipu.MODEL_PREFIX):
        return zhipu.generate_stream(prompt, model)
    return generate_ollama_prompt_stream(prompt, model=model)


def _build_run_config(*, route_mode: str, route_reason: str, model: str) -> dict:
    """构造本轮问答的在线配置快照（路线图 M3.5-④）。

    落库到 chat_turns.run_config，让一次回答能关联到它使用的在线模型配置。
    **隐私红线**（写在这里是因为调用方都从这里拿快照）：
    - 不得出现任何 API Key / 密钥（模型名只是名字，不含凭据）；
    - 不得包含小说原文片段或检索内容——原文证据已经在 sources 列单独落库，
      快照只记录"用了什么配置"，不复制"看到了什么内容"；
    - 除路由原因这类短描述外，不得携带用户输入的其他隐私。
    """
    return {
        # prompt 模板版本：模板文本改动必须递增 generation_mixin.PROMPT_TEMPLATE_VERSION
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        # 回答模式与路由决策原因（auto/grounded/free + 为什么走了这条路）
        "answer_mode": route_mode,
        "route_reason": route_reason,
        # 生成模型（当前回答用的那个）
        "generate_model": model,
        # 重排配置：模型名 + 开关
        "rerank_enabled": RERANK_ENABLED,
        "reranker_model": RERANKER_MODEL if RERANK_ENABLED else None,
        # 查询改写/扩展是否可能参与本轮检索
        "query_rewrite_enabled": QUERY_REWRITE_ENABLED,
        "query_expand_enabled": QUERY_EXPAND_ENABLED,
    }


@app.post("/api/ask")
async def ask(req: AskRequest, request: Request):
    """流式回答。支持用户中断：前端 abort 后，这里会停止向上游模型要 token。

    刻意写成 async def（而不是同步 def）：只有在协程里才能 await 出让控制权，
    从而定期检查 `request.is_disconnected()`。写成同步函数时 FastAPI 会丢进线程池，
    客户端断开后那个线程仍会把生成跑到底——白烧本地 GPU，或继续消耗用户的
    Claude/GLM 付费额度。这不是优化，是避免"用户点了停止还在扣他的钱"。
    """
    decision = choose_answer_route(req.question, req.mode)
    rag: NovelRAG | None = state.get("rag")
    if decision.route is AnswerMode.grounded and rag is None:
        raise APIError(409, ErrorCode.index_not_ready, "书架为空或索引未建立，请先上传小说")

    # 多轮改写：把"他后来怎么样了"这类带指代的追问补全成独立完整的问题，
    # 再拿补全后的问题去检索。**只影响检索**——存库、显示、送给模型的都还是
    # 用户原始问题（见下面 save_turn 用的是 req.question）。
    # 自由问答不检索，所以也没必要额外花一次模型调用做“面向检索”的问题改写。
    # 历史读一次，供查询改写和回答 prompt 共用（M3.6）
    history = _load_history(req.session_id) if decision.route is AnswerMode.grounded else []
    search_question = (
        _rewrite_for_search(req, history)
        if decision.route is AnswerMode.grounded
        else req.question
    )

    model = state["model"]
    # M3.5-④：在线配置快照在生成前定格——它描述"这轮回答用了什么配置"，
    # 不随生成成败变化；最终状态（complete/interrupted/error）落库时才补上。
    run_config = _build_run_config(
        route_mode=decision.route.value, route_reason=decision.reason, model=model
    )

    # 有 session_id 就落库，便于刷新页面后恢复历史；没有就纯内存、行为跟以前一致。
    session_id = req.session_id
    user_index = assistant_index = None
    if session_id:
        try:
            user_index = next_turn_index(session_id)
            assistant_index = user_index + 1
            save_turn(session_id, user_index, "user", req.question)
        except Exception as exc:  # 落库失败不该让提问功能不可用
            logger.warning(f"保存提问失败（忽略，不影响回答）：{exc}")
            session_id = None

    async def event_stream():
        # ----------------------------------------------------- 路由与检索阶段
        # 检索放在流里逐步跑，有两个原因：
        #
        # 1）**不阻塞事件循环**。之前是在这个 async def 里同步调用检索，
        #    整整 2 秒（交叉编码器重排占大头）没有任何 await——期间整个服务
        #    的其他请求全部卡住，连自己的断连检查都跑不了。写成 async def
        #    却在里面跑同步重活，等于白写。
        # 2）**让用户看见进度**。原来 5 个步骤要等检索全跑完才一次性弹出，
        #    前 2 秒界面上什么都没有。现在每完成一步就推一条，界面可以像
        #    成熟的 AI 应用那样把步骤一条条点亮。等待时长没变，但心理感受完全不同。
        trace_payload: list[dict] = []
        route_step = TraceStep(
            step="回答路径",
            detail=(
                f"{decision.reason}，将检索小说原文后回答"
                if decision.route is AnswerMode.grounded
                else f"{decision.reason}，将直接使用模型回答，不搜索小说"
            ),
        ).model_dump()
        trace_payload.append(route_step)
        yield f"event: step\ndata: {json.dumps(route_step, ensure_ascii=False)}\n\n"

        if search_question != req.question:
            # 让用户在「思考过程」里看得见改写发生了——改写是个会改变检索结果的
            # 隐式步骤，不展示出来的话，用户无法理解"为什么搜出了这些"。
            first = TraceStep(
                step="理解追问", detail=f"补全指代后按「{search_question}」检索"
            ).model_dump()
            trace_payload.append(first)
            yield f"event: step\ndata: {json.dumps(first, ensure_ascii=False)}\n\n"

        if HISTORY_IN_PROMPT and history:
            # 「每次压缩都能在 trace 中解释」是 M3.6 的验收要求之一：带了几轮、
            # 丢了什么、为什么丢，都要能在界面上看到，而不是变成一个黑箱。
            _, history_trace = build_history_block(history)
            history_step = TraceStep(
                step="对话背景",
                detail=(
                    f"带入最近 {history_trace['turns_used']}/"
                    f"{history_trace['turns_available']} 轮对话"
                    f"（{history_trace['chars']} 字）：{history_trace['reason']}"
                ),
            ).model_dump()
            trace_payload.append(history_step)
            yield f"event: step\ndata: {json.dumps(history_step, ensure_ascii=False)}\n\n"

        sources = []
        context_sources = []
        structured_answer: str | None = None
        if decision.route is AnswerMode.grounded:
            # 上面已经保证 grounded 路径下 rag 不为 None；assert 也让类型和不变量
            # 对阅读代码的人保持一致，而不是在后面散落一串 if rag。
            assert rag is not None
            structured_answer = getattr(rag, "library_answer", lambda _q: None)(
                search_question
            )
            if structured_answer is not None:
                # 目录问题使用完整数据库事实，不能让 top-k 召回范围决定答案。
                structured_step = TraceStep(
                    step="结构化查询",
                    detail="问题需要书架完整目录，已跳过局部片段召回并直接读取数据库元数据",
                    ms=0,
                ).model_dump()
                trace_payload.append(structured_step)
                yield f"event: step\ndata: {json.dumps(structured_step, ensure_ascii=False)}\n\n"
            else:
                step_iter = rag.retrieve_hybrid_stream(search_question, top_k=req.top_k)
                while True:
                    # 和下面消费模型 token 用的是同一套模式：同步生成器丢线程池里逐个取，
                    # 每个 await 都是一次让出控制权的机会。
                    item = await run_in_threadpool(_next_or_sentinel, step_iter)
                    if item is _SENTINEL:
                        break
                    kind, value = item
                    if kind == "result":
                        sources = value
                        continue
                    # 过一遍 Pydantic 模型再转回 dict：StreamingResponse 不支持声明
                    # response_model，这里手动保证发出去和存进库的形状不会手滑写错字段。
                    payload_step = TraceStep(**value).model_dump()
                    trace_payload.append(payload_step)
                    yield f"event: step\ndata: {json.dumps(payload_step, ensure_ascii=False)}\n\n"

                context_sources, expand_step = rag.build_answer_context(sources)
                if expand_step is not None:
                    # chapter 实验档才会走到这里：把模式、证据 token 数、是否截断
                    # 并入「思考过程」，评测复盘时能解释"模型到底看到了什么"。
                    expand_payload = TraceStep(**expand_step).model_dump()
                    trace_payload.append(expand_payload)
                    yield (
                        "event: step\n"
                        f"data: {json.dumps(expand_payload, ensure_ascii=False)}\n\n"
                    )
        # 引用编号必须与模型看到的 context_sources 一一对应。之前只把最初 top-k
        # 发给前端、邻居片段只给模型；加了 [n] 后那会导致模型引用 [4]，界面却只有
        # 3 张卡片。现在 SSE、历史记录和 prompt 共用同一份最终上下文来源。
        payload = [
            SourceItem(
                novel=s.novel,
                chunk_id=s.chunk_id,
                chapter_title=getattr(s, "chapter_title", None),
                text=s.text,
            ).model_dump()
            for s in context_sources
        ]
        # 把最终上下文来源发出，前端可立即渲染并接受 [n] 定位
        yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 逐 token 流式推送答案：按模型名前缀路由到对应的生成后端
        # claude:xxx → 本地 Claude Code CLI（用户自己的订阅）
        # glm:xxx    → 智谱开放平台（用户自己的 ZHIPU_API_KEY）
        # 其余       → 本地 Ollama
        if structured_answer is not None:
            # 结构化目录事实是确定结果，不再让模型从局部片段中猜测。
            token_iter = iter([structured_answer])
        else:
            prompt = (
                rag.build_prompt(
                    req.question,
                    context_sources,
                    history=history if HISTORY_IN_PROMPT else None,
                )
                if decision.route is AnswerMode.grounded and rag is not None
                else build_free_prompt(req.question)
            )
            # grounded 和 free 都使用已经构造好的 prompt，避免 grounded 本地路径
            # 再 build_prompt 一次（图线索查询等工作也会被重复执行）。
            token_iter = _generate_for_model(prompt, model)

        parts: list[str] = []
        interrupted = False
        error: str | None = None
        try:
            while True:
                # 三个生成后端都是同步生成器，直接在协程里 for 会阻塞事件循环、
                # 导致下面的断连检查永远等不到机会执行。放线程池里逐个取，
                # 每个 await 都是一次让出控制权的机会。
                chunk = await run_in_threadpool(_next_or_sentinel, token_iter)
                if chunk is _SENTINEL:
                    break
                if not chunk:
                    continue
                parts.append(chunk)
                yield f"event: token\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                # 协作式取消：不强杀线程，而是发现客户端走了就自己收手，
                # 并且关掉生成器（close() 会让上游 requests/subprocess 连接断开，
                # 停止继续向 Ollama/GLM 索取 token）。
                if await request.is_disconnected():
                    interrupted = True
                    break
        except Exception as exc:
            # M3.5-④：生成中途抛错时如实记录 error 状态（此前 finally 里一律
            # 存 complete，中断和异常无法区分）。不吞掉异常，继续向上传播。
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if interrupted:
                token_iter.close()
            if session_id and assistant_index is not None:
                final_status = (
                    "interrupted" if interrupted else ("error" if error else "complete")
                )
                try:
                    save_turn(
                        session_id,
                        assistant_index,
                        "assistant",
                        "".join(parts),
                        sources=payload,
                        trace=trace_payload,
                        status=final_status,
                        # 最终状态同时并入快照，让 run_config 自包含可追溯；
                        # error_message 只记异常类型+摘要，不含用户输入原文
                        run_config={
                            **run_config,
                            "final_status": final_status,
                            **({"error": error[:200]} if error else {}),
                        },
                    )
                except Exception as exc:
                    logger.warning(f"保存回答失败（忽略）：{exc}")

        if not interrupted:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/agent/ask")
async def agent_ask(req: AgentAskRequest, request: Request):
    """运行独立 Agent Lab；最多五步，只能调用白名单只读工具。"""
    rag: NovelRAG | None = state.get("rag")
    if rag is None:
        raise APIError(409, ErrorCode.index_not_ready, "书架为空或索引未建立，请先上传小说")
    model = state["model"]

    def planner(prompt: str) -> str:
        # 普通回答可以边生成边展示；工具规划不同，必须先拿到完整 JSON 才能校验
        # tool/args，不能收到半个对象就执行。这里仍复用同一模型适配器，只在边界
        # 处把 token 流合并成一次 action。
        return "".join(_generate_for_model(prompt, model))

    iterator = run_agent(
        req.question,
        rag=rag,
        planner=planner,
        answerer=lambda prompt: _generate_for_model(prompt, model),
        max_steps=req.max_steps,
    )
    # M3.5-③：本次 Agent 运行的 run_id。request_id 只覆盖单次 HTTP 请求，而
    # Agent 的一次运行可能横跨多个请求（未来的重试/恢复场景），所以单独发一个
    # 轻量 id 注入每个 agent_step——同一次运行的所有步骤共享同一个值，
    # 落库后可以按它还原完整事件顺序。不重构现有事件结构，只加一个可选字段。
    run_id = uuid.uuid4().hex[:12]

    # 和 /api/ask 同一套模式：有 session_id 才落库，没有就纯内存、行为不变。
    # 这个端点上线时漏了这一步——Agent Lab 里的每一次对话都不会落库，
    # 刷新页面必然清空，跟普通问答模式的历史恢复体验不一致。
    session_id = req.session_id
    user_index = assistant_index = None
    if session_id:
        try:
            user_index = next_turn_index(session_id)
            assistant_index = user_index + 1
            save_turn(session_id, user_index, "user", req.question)
        except Exception as exc:  # 落库失败不该让 Agent Lab 不可用
            logger.warning(f"保存提问失败（忽略，不影响回答）：{exc}")
            session_id = None

    async def event_stream():
        # Agent 的 Python 循环是同步生成器，放进线程池逐步 next，避免模型规划、
        # 数据库工具或最终生成阻塞 FastAPI 事件循环。客户端断开时 close() 会让
        # 生成器 finally 生效，与普通问答的“停止生成”保持相同资源边界。
        agent_steps_payload: list[dict] = []
        sources_payload: list[dict] = []
        parts: list[str] = []
        interrupted = False
        try:
            while True:
                item = await run_in_threadpool(_next_or_sentinel, iterator)
                if item is _SENTINEL:
                    break
                kind, value = item
                if kind == "agent_step":
                    payload = AgentStep(run_id=run_id, **value).model_dump()
                    agent_steps_payload.append(payload)
                    yield f"event: agent_step\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif kind == "sources":
                    sources_payload = [
                        SourceItem(
                            novel=source.novel,
                            chunk_id=source.chunk_id,
                            chapter_title=source.chapter_title,
                            text=source.text,
                        ).model_dump()
                        for source in value
                    ]
                    yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"
                elif kind == "token":
                    parts.append(value)
                    yield f"event: token\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"
                elif kind == "done":
                    yield "event: done\ndata: {}\n\n"
                if await request.is_disconnected():
                    interrupted = True
                    iterator.close()
                    break
        finally:
            iterator.close()
            if session_id and assistant_index is not None:
                try:
                    save_turn(
                        session_id,
                        assistant_index,
                        "assistant",
                        "".join(parts),
                        sources=sources_payload,
                        agent_steps=agent_steps_payload,
                        status="interrupted" if interrupted else "complete",
                    )
                except Exception as exc:
                    logger.warning(f"保存回答失败（忽略）：{exc}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ----------------------------------------------------------------- 按需核实引用
@app.post("/api/citations/verify", response_model=VerifyCitationResult)
async def verify_citation(req: VerifyCitationRequest):
    """核实某一条 ``[n]`` 引用是否真的被它指向的原文支持。

    **为什么做成用户点一下才跑，而不是每次回答自动核实**：

    `docs/experiments/m35-faithfulness-calibration.md` 给忠实度判定定了两档启用
    门槛，最轻的一档（UI 显示"忠实度不确定"提示）要求一致率 ≥80%，而实测最好的
    配置只有 67.9%，所以自动、全局地给回答打忠实度标记目前不达标——尤其是
    "反对"方向的精确率只有 50%，自动告警每两次就有一次冤枉，比不提示更糟。

    按需核实绕开的正是这个问题：判定由用户主动发起、只作用于他指定的那一条引用，
    返回的是**原始判定和理由**（连同判定用的模型名），而不是系统给整段回答盖章。
    用户自己能看着理由决定信不信，这和那张阈值表管的"自动标注"是两回事。

    成本也因此可控：不点不花钱、不增加任何一次回答的延迟。
    """
    statements = statements_citing(req.answer, req.citation)
    if not statements:
        raise APIError(
            400,
            ErrorCode.validation_error,
            f"回答里没有引用 [{req.citation}]，无法核实",
        )
    model = state["model"]
    # 判定本身是同步阻塞的模型调用（实测单次 10~60s），放线程池避免卡住事件循环
    result = await run_in_threadpool(
        judge_support,
        "".join(statements),
        req.evidence,
        lambda prompt: _generate_for_model(prompt, model),
    )
    return VerifyCitationResult(
        label=result["label"],
        reason=result["reason"],
        statement="".join(statements),
        model=model,
    )


# ----------------------------------------------------------------- 会话历史
@app.get("/api/sessions/{session_id}", response_model=SessionHistory)
def get_session(session_id: str):
    """读回某个会话的全部对话，用于刷新页面后恢复界面。"""
    try:
        rows = load_turns(session_id)
    except Exception as exc:
        raise APIError(500, ErrorCode.session_read_failed, f"读取会话失败：{exc}") from exc
    return SessionHistory(session_id=session_id, turns=[_restore_turn(row) for row in rows])


# 结构化附加字段：校验失败时可以单独丢掉而不影响这一轮的正文内容
_OPTIONAL_TURN_PAYLOADS = ("agent_steps", "trace", "sources")


def _restore_turn(row: dict) -> StoredTurn:
    """把一行历史记录还原成 StoredTurn，结构化字段坏掉时降级而不是整体失败。

    为什么需要降级：``sources``/``trace``/``agent_steps`` 是 jsonb 列，形状由写入
    当时的代码版本决定，而这些形状**一直在演进**（TraceStep 先后加过 ms /
    stage_key / candidates；chat_turns 后来才加 agent_steps）。一旦某次改动让旧
    记录不再满足新模型，`[StoredTurn.model_validate(row) for row in rows]` 会在
    第一条坏行上抛异常，导致整个会话的历史全部读不出来——**用户丢掉的不是那
    一轮的步骤卡片，而是整段对话**。

    正文（role/content/status）才是用户真正在乎的东西，且形状稳定。所以坏掉的
    只丢结构化附加字段：对话仍然完整可读，只是那一轮不再显示出处或步骤轨迹。
    """
    try:
        return StoredTurn.model_validate(row)
    except ValidationError:
        degraded = {**row, **dict.fromkeys(_OPTIONAL_TURN_PAYLOADS)}
        try:
            turn = StoredTurn.model_validate(degraded)
        except ValidationError:
            # 连正文都不合法（role/content/status 缺失或类型不对）说明这行是真的
            # 坏了，不是形状演进问题——此时才让它冒泡成 500，不静默吞掉。
            raise
        logger.warning(
            "会话 %s 第 %s 轮的结构化字段无法解析，已降级为纯文本恢复",
            row.get("session_id", "?"),
            row.get("turn_index", "?"),
        )
        return turn


# ----------------------------------------------------------------- 关系边审核（M4）
@app.get("/api/graph/edges", response_model=GraphEdgeList)
def list_graph_edges(
    status: str = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """分页列出人物关系边，供审核面板消费。默认只看 pending（待审核队列）。

    status 可选 pending/approved/rejected；传 all 列出全部状态。
    """
    if status != "all" and status not in VALID_REVIEW_STATUSES:
        raise APIError(400, ErrorCode.validation_error, f"未知的审核状态：{status}")
    try:
        edges, total = list_relation_edges(
            status=None if status == "all" else status, limit=limit, offset=offset
        )
    except Exception as exc:
        raise APIError(500, ErrorCode.internal_error, f"读取关系边失败：{exc}") from exc
    # dict 行显式过一遍 Pydantic：字段缺失/类型漂移在这里报错，而不是序列化时
    return GraphEdgeList(
        total=total,
        limit=limit,
        offset=offset,
        edges=[GraphEdgeItem.model_validate(edge) for edge in edges],
    )


@app.post("/api/graph/review", response_model=GraphReviewResult)
def review_graph_edge(req: GraphReviewRequest):
    """写入一条关系边的人工审核结论。

    rejected 的边在所有查询里立即不可见（可见性过滤见 postgres.query_relations）；
    approved 的 co_occurrence 边即使开着「只要明确陈述」的门槛也不会自动进入
    在线结果——门槛过滤的是 evidence_type，审核通过解决的是"这条共现边我看过，
    是真的"。两者语义不同，刻意不混用。
    """
    if req.status not in {"approved", "rejected"}:
        raise APIError(
            400,
            ErrorCode.validation_error,
            "审核动作只能是 approved（通过）或 rejected（拒绝）",
        )
    try:
        updated = set_relation_review(
            req.novel,
            req.person_a,
            req.person_b,
            req.relation,
            req.status,
        )
    except Exception as exc:
        raise APIError(500, ErrorCode.internal_error, f"写入审核结论失败：{exc}") from exc
    if updated == 0:
        raise APIError(404, ErrorCode.book_not_found, "关系边不存在（可能已被重建索引刷新）")
    return GraphReviewResult(review_status=req.status)


# ----------------------------------------------------------------- 模型切换
def _list_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []  # Ollama 没跑起来也不阻塞——Claude 选项照样可用


def _available_models() -> list[str]:
    """本地 Ollama 已安装的 + 装了 claude CLI 时的 Claude 订阅 + 配了 ZHIPU_API_KEY 时的 GLM。"""
    return _list_ollama_models() + claude_cli.claude_model_options() + zhipu.model_options()


@app.get("/api/models", response_model=ModelList)
def list_models():
    return ModelList(models=_available_models(), current=state["model"])


@app.post("/api/model", response_model=CurrentModel)
def set_model(req: SetModelRequest):
    if req.model not in _available_models():
        raise APIError(400, ErrorCode.model_unavailable, f"模型 {req.model} 当前不可用")
    state["model"] = req.model
    return CurrentModel(current=state["model"])


@app.get("/api/health", response_model=HealthStatus)
def health():
    return HealthStatus(ok=True, ready=state.get("rag") is not None)


# ------------------------------------------------------------- 前端静态托管（生产）
# frontend/dist 存在时由 FastAPI 直接托管前端产物，单端口对外（Docker 镜像即此形态，
# 见 README「生产部署」）。挂载必须放在所有 API 路由之后：Starlette 按注册顺序匹配，
# "/" 兜底不影响 /api/*。html=True 让未命中的路径回落 index.html，SPA 刷新不 404。
# 开发模式下 dist 不存在（Vite dev server 自己服务前端），这个分支整体不生效。
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
