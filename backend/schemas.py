"""API 请求/响应的 Pydantic 模型，集中放这里而不是散在 main.py 里。

集中的好处：FastAPI 据此生成的 OpenAPI 契约才是完整的（之前大部分端点直接
返回裸 dict，/docs 里看到的响应体全是空的 {}）；前端要是想从 OpenAPI 生成
TS 类型，这里就是唯一真源，不用再靠人肉对着 main.py 的返回语句猜字段。

注意：本模块依赖 `backend/main.py` 顶部已经把 src/ 加进 sys.path
（`from config import TOP_K` 才能找到）——不要在别处独立导入这个模块。
"""
from pydantic import BaseModel

from config import TOP_K


# ----------------------------------------------------------------- 请求体
class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    # 可选：带上会话 ID 就把这一轮问答落库，刷新页面后能恢复。
    # 不传则完全不落库，行为跟以前一致（纯内存对话）。
    session_id: str | None = None


class SetModelRequest(BaseModel):
    model: str


# ----------------------------------------------------------------- 书架
class BookList(BaseModel):
    books: list[str]


class ReindexResult(BaseModel):
    novels: list[str]
    chunk_count: int


class UploadResult(ReindexResult):
    saved: list[str]


class DeleteResult(ReindexResult):
    deleted: str


# ----------------------------------------------------------------- 全文搜索
class SearchMatch(BaseModel):
    novel: str
    chunk_id: int
    text: str
    match_count: int


class SearchResult(BaseModel):
    query: str
    total: int
    results: list[SearchMatch]


# ----------------------------------------------------------------- 问答（SSE 事件 payload）
# 这两个是 /api/ask 流式响应里 event: trace / event: sources 各自的 data 形状。
# StreamingResponse 本身不支持声明 response_model（FastAPI 不会校验流式响应体），
# 但构造这些 JSON 时经过模型再 model_dump()，至少能保证字段名和类型不会手滑写错。
class TraceStep(BaseModel):
    step: str
    detail: str
    # 本阶段耗时（毫秒）。可选：历史会话里存的旧记录没有这个字段。
    ms: int | None = None


class SourceItem(BaseModel):
    novel: str
    chunk_id: int
    text: str


# ----------------------------------------------------------------- 会话历史
class StoredTurn(BaseModel):
    turn_index: int
    role: str
    content: str
    sources: list[SourceItem] | None = None
    trace: list[TraceStep] | None = None
    status: str


class SessionHistory(BaseModel):
    session_id: str
    turns: list[StoredTurn]


# ----------------------------------------------------------------- 模型切换
class ModelList(BaseModel):
    models: list[str]
    current: str


class CurrentModel(BaseModel):
    current: str


# ----------------------------------------------------------------- 健康检查
class HealthStatus(BaseModel):
    ok: bool
    ready: bool
