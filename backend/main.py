"""FastAPI 后端：把 src 下的 RAG 逻辑包成 HTTP 接口。

运行：uvicorn backend.main:app --reload --port 8000
（在项目根目录 novel-rag/ 下运行）
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 让后端能 import src 下的业务逻辑（完全复用，不改动）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backend import claude_cli  # noqa: E402
import ingest  # noqa: E402
from config import NOVELS_DIR, OLLAMA_HOST, OLLAMA_MODEL, TOP_K  # noqa: E402
from rag import NovelRAG  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from config import EMBEDDING_MODEL  # noqa: E402

# 进程级共享资源（对应 Streamlit 的 cache_resource）
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载一次 embedding 模型，并尝试连接向量库
    state["embedder"] = SentenceTransformer(EMBEDDING_MODEL)
    state["rag"] = _try_load_rag()
    state["model"] = OLLAMA_MODEL  # 当前用于生成回答的模型，可通过 /api/model 动态切换
    yield
    state.clear()


def _try_load_rag() -> NovelRAG | None:
    try:
        return NovelRAG(embedder=state["embedder"])
    except Exception:
        return None  # 向量库还没建立


app = FastAPI(title="书虫 · Novel RAG API", lifespan=lifespan)

# 开发期允许 Vite dev server 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- 数据模型
class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class BookList(BaseModel):
    books: list[str]


class SetModelRequest(BaseModel):
    model: str


# ----------------------------------------------------------------- 书架
@app.get("/api/books", response_model=BookList)
def list_books():
    return BookList(books=sorted(p.stem for p in NOVELS_DIR.glob("*.txt")))


@app.post("/api/books")
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


@app.delete("/api/books/{name}")
def delete_book(name: str):
    # 只允许删除 novels 目录下的 txt，拒绝路径穿越
    target = (NOVELS_DIR / f"{Path(name).name}.txt").resolve()
    if target.parent != NOVELS_DIR.resolve() or not target.exists():
        raise HTTPException(404, "书不存在")
    target.unlink()
    result = _reindex()
    return {"deleted": name, **result}


@app.post("/api/reindex")
def reindex():
    return _reindex()


def _reindex() -> dict:
    result = ingest.build_index(model=state["embedder"])
    # 重建后刷新 RAG 句柄，使新库生效
    state["rag"] = _try_load_rag() if result["chunk_count"] else None
    return result


# ----------------------------------------------------------------- 提问（SSE 流式）
@app.post("/api/ask")
def ask(req: AskRequest):
    rag: NovelRAG | None = state.get("rag")
    if rag is None:
        raise HTTPException(409, "书架为空或索引未建立，请先上传小说")

    sources = rag.retrieve(req.question, top_k=req.top_k)
    model = state["model"]

    def event_stream():
        # 先把来源作为一个事件发出，前端可立即渲染出处
        payload = [
            {"novel": s.novel, "chunk_id": s.chunk_id, "text": s.text}
            for s in sources
        ]
        yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        # 再逐 token 流式推送答案：claude:xxx 走本地 Claude Code CLI（用户自己的订阅），
        # 其余走本地 Ollama
        if model.startswith(claude_cli.MODEL_PREFIX):
            prompt = rag.build_prompt(req.question, sources)
            token_iter = claude_cli.generate_stream(prompt, model)
        else:
            token_iter = rag.generate_stream(req.question, sources, model=model)
        for chunk in token_iter:
            yield f"event: token\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ----------------------------------------------------------------- 模型切换
def _list_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []  # Ollama 没跑起来也不阻塞——Claude 选项照样可用


@app.get("/api/models")
def list_models():
    """列出可选模型：本地 Ollama 已安装的 + （如果本机装了 claude CLI）用户自己的 Claude 订阅。"""
    models = _list_ollama_models() + claude_cli.claude_model_options()
    return {"models": models, "current": state["model"]}


@app.post("/api/model")
def set_model(req: SetModelRequest):
    available = _list_ollama_models() + claude_cli.claude_model_options()
    if req.model not in available:
        raise HTTPException(400, f"模型 {req.model} 当前不可用")
    state["model"] = req.model
    return {"current": state["model"]}


@app.get("/api/health")
def health():
    return {"ok": True, "ready": state.get("rag") is not None}
