"""FastAPI 后端：把 src 下的 RAG 逻辑包成 HTTP 接口。

运行：uvicorn backend.main:app --reload --port 8000
（在项目根目录 novel-rag/ 下运行）
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

from backend import claude_cli, zhipu  # noqa: E402
from backend.middleware import RequestIDMiddleware  # noqa: E402
from backend.schemas import (  # noqa: E402
    AskRequest,
    BookList,
    CurrentModel,
    DeleteResult,
    HealthStatus,
    ModelList,
    ReindexResult,
    SearchMatch,
    SearchResult,
    SessionHistory,
    SetModelRequest,
    SourceItem,
    StoredTurn,
    TraceStep,
    UploadResult,
)
import ingest  # noqa: E402
from config import NOVELS_DIR, OLLAMA_HOST, OLLAMA_MODEL  # noqa: E402
from rag import NovelRAG  # noqa: E402
from loader import load_novel_chunks  # noqa: E402
from postgres import (  # noqa: E402
    close_pool,
    connect,
    ensure_chat_schema,
    has_index,
    init_pool,
    load_turns,
    next_turn_index,
    save_turn,
)
from embedder import load_embedder  # noqa: E402

# 进程级共享资源（对应 Streamlit 的 cache_resource）
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 只打变量名不打值：既能确认密钥已加载，又不会把密钥写进日志
    if _ENV_KEYS:
        logger.info(f"已从 .env 加载：{', '.join(_ENV_KEYS)}")
    logger.info(
        f"云端可选模型：{claude_cli.claude_model_options() + zhipu.model_options()}"
    )
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
    # 启动时加载一次 embedding 模型，并尝试连接 PostgreSQL 索引
    state["embedder"] = load_embedder()
    state["rag"] = _try_load_rag()
    state["chunks"] = load_novel_chunks(NOVELS_DIR)
    state["model"] = OLLAMA_MODEL  # 当前用于生成回答的模型，可通过 /api/model 动态切换
    yield
    state.clear()
    close_pool()


def _try_load_rag() -> NovelRAG | None:
    try:
        return NovelRAG(embedder=state["embedder"])
    except Exception:
        return None  # PostgreSQL 索引还没建立


app = FastAPI(title="书虫 · Novel RAG API", lifespan=lifespan)

# 开发期允许 Vite dev server 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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


@app.post("/api/books", response_model=UploadResult)
async def upload_books(files: list[UploadFile]):
    saved = []
    for f in files:
        name = Path(f.filename or "").name  # 防止路径穿越
        if not name.lower().endswith(".txt"):
            continue
        (NOVELS_DIR / name).write_bytes(await f.read())
        saved.append(Path(name).stem)
    if not saved:
        raise HTTPException(400, "没有有效的 .txt 文件")
    result = _reindex()
    return {"saved": saved, **result}


@app.delete("/api/books/{name}", response_model=DeleteResult)
def delete_book(name: str):
    # 只允许删除 novels 目录下的 txt，拒绝路径穿越
    target = (NOVELS_DIR / f"{Path(name).name}.txt").resolve()
    if target.parent != NOVELS_DIR.resolve() or not target.exists():
        raise HTTPException(404, "书不存在")
    target.unlink()
    result = _reindex()
    return {"deleted": name, **result}


@app.post("/api/reindex", response_model=ReindexResult)
def reindex():
    return _reindex()


def _reindex() -> dict:
    result = ingest.build_index(model=state["embedder"])
    state["chunks"] = load_novel_chunks(NOVELS_DIR)
    # 重建后刷新 RAG 句柄，使新库生效
    state["rag"] = _try_load_rag() if result["chunk_count"] else None
    return result


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
        raise HTTPException(409, "PostgreSQL 索引未建立，请先重新整理书架")

    if book:
        where_sql = "novel = %s AND position(lower(%s) in lower(text)) > 0"
        query_params = (book, needle)
    else:
        where_sql = "position(lower(%s) in lower(text)) > 0"
        query_params = (needle,)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT novel, chunk_id, text
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
_SENTINEL = object()  # 线程池里取不到下一个 token 时的哨兵，区别于"取到了 None"


def _next_or_sentinel(iterator):
    """在线程里取生成器的下一个元素；取完返回哨兵而不是抛 StopIteration。

    StopIteration 不能穿过 await 边界（会变成 RuntimeError），所以用哨兵传递结束。
    """
    return next(iterator, _SENTINEL)


@app.post("/api/ask")
async def ask(req: AskRequest, request: Request):
    """流式回答。支持用户中断：前端 abort 后，这里会停止向上游模型要 token。

    刻意写成 async def（而不是同步 def）：只有在协程里才能 await 出让控制权，
    从而定期检查 `request.is_disconnected()`。写成同步函数时 FastAPI 会丢进线程池，
    客户端断开后那个线程仍会把生成跑到底——白烧本地 GPU，或继续消耗用户的
    Claude/GLM 付费额度。这不是优化，是避免"用户点了停止还在扣他的钱"。
    """
    rag: NovelRAG | None = state.get("rag")
    if rag is None:
        raise HTTPException(409, "书架为空或索引未建立，请先上传小说")

    # 同时召回关键词、语义、结构性片段，合并排序后统一交给模型回答。
    # trace 记录每一步的真实动作，前端展示为可折叠的「思考过程」。
    sources, trace = rag.retrieve_hybrid_traced(req.question, top_k=req.top_k)
    context_sources = rag.expand_neighbors(sources)
    model = state["model"]
    # 过一遍 Pydantic 模型再转回 dict：StreamingResponse 本身不支持声明
    # response_model，这里手动保证发到前端和存进数据库的形状不会手滑写错字段。
    trace_payload = [TraceStep(**t).model_dump() for t in trace]
    payload = [
        SourceItem(novel=s.novel, chunk_id=s.chunk_id, text=s.text).model_dump()
        for s in sources
    ]

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
        # 先把「思考过程」发出去，让用户在等生成时就能看到检索是怎么做的
        yield f"event: trace\ndata: {json.dumps(trace_payload, ensure_ascii=False)}\n\n"
        # 再把来源发出，前端可立即渲染出处
        yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 逐 token 流式推送答案：按模型名前缀路由到对应的生成后端
        # claude:xxx → 本地 Claude Code CLI（用户自己的订阅）
        # glm:xxx    → 智谱开放平台（用户自己的 ZHIPU_API_KEY）
        # 其余       → 本地 Ollama
        if model.startswith(claude_cli.MODEL_PREFIX):
            token_iter = claude_cli.generate_stream(
                rag.build_prompt(req.question, context_sources), model
            )
        elif model.startswith(zhipu.MODEL_PREFIX):
            token_iter = zhipu.generate_stream(
                rag.build_prompt(req.question, context_sources), model
            )
        else:
            token_iter = rag.generate_stream(req.question, context_sources, model=model)

        parts: list[str] = []
        interrupted = False
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
        finally:
            if interrupted:
                token_iter.close()
            if session_id and assistant_index is not None:
                try:
                    save_turn(
                        session_id,
                        assistant_index,
                        "assistant",
                        "".join(parts),
                        sources=payload,
                        trace=trace_payload,
                        status="interrupted" if interrupted else "complete",
                    )
                except Exception as exc:
                    logger.warning(f"保存回答失败（忽略）：{exc}")

        if not interrupted:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ----------------------------------------------------------------- 会话历史
@app.get("/api/sessions/{session_id}", response_model=SessionHistory)
def get_session(session_id: str):
    """读回某个会话的全部对话，用于刷新页面后恢复界面。"""
    try:
        turns = load_turns(session_id)
    except Exception as exc:
        raise HTTPException(500, f"读取会话失败：{exc}") from exc
    return SessionHistory(session_id=session_id, turns=turns)


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
    return (
        _list_ollama_models()
        + claude_cli.claude_model_options()
        + zhipu.model_options()
    )


@app.get("/api/models", response_model=ModelList)
def list_models():
    return ModelList(models=_available_models(), current=state["model"])


@app.post("/api/model", response_model=CurrentModel)
def set_model(req: SetModelRequest):
    if req.model not in _available_models():
        raise HTTPException(400, f"模型 {req.model} 当前不可用")
    state["model"] = req.model
    return CurrentModel(current=state["model"])


@app.get("/api/health", response_model=HealthStatus)
def health():
    return HealthStatus(ok=True, ready=state.get("rag") is not None)
