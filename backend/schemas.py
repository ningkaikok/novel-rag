"""API 请求/响应的 Pydantic 模型，集中放这里而不是散在 main.py 里。

集中的好处：FastAPI 据此生成的 OpenAPI 契约才是完整的（之前大部分端点直接
返回裸 dict，/docs 里看到的响应体全是空的 {}）；前端要是想从 OpenAPI 生成
TS 类型，这里就是唯一真源，不用再靠人肉对着 main.py 的返回语句猜字段。

注意：本模块依赖 `backend/main.py` 顶部已经把 src/ 加进 sys.path
（`from config import TOP_K` 才能找到）——不要在别处独立导入这个模块。
"""

from pydantic import BaseModel, Field

from config import TOP_K
from query_router import AnswerMode


# ----------------------------------------------------------------- 请求体
class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    # 可选：带上会话 ID 就把这一轮问答落库，刷新页面后能恢复。
    # 不传则完全不落库，行为跟以前一致（纯内存对话）。
    session_id: str | None = None
    # auto：后端保守判断；grounded：强制查小说；free：不查小说、直接问模型。
    mode: AnswerMode = AnswerMode.auto


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    max_steps: int = Field(default=5, ge=3, le=5)
    # 可选：带上会话 ID 就把这一轮 Agent 对话落库，刷新页面后能恢复。
    # 之前这个端点完全没有这个字段——Agent Lab 里的每一次对话都是纯内存，
    # 刷新页面必然清空，跟普通问答模式的历史恢复体验不一致。
    session_id: str | None = None


class SetModelRequest(BaseModel):
    model: str


# ----------------------------------------------------------------- 书架
class BookList(BaseModel):
    books: list[str]


class IndexResult(BaseModel):
    novels: list[str]
    chunk_count: int
    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    contextualized: int = 0
    relations: int = 0
    hierarchy_nodes: int = 0


class IndexTaskStatus(BaseModel):
    id: str
    status: str
    stage: str
    progress: int
    message: str
    error: str | None = None
    force: bool = False
    retry_of: str | None = None
    result: IndexResult | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class UploadResult(BaseModel):
    saved: list[str]
    task: IndexTaskStatus


class DeleteResult(BaseModel):
    deleted: str
    task: IndexTaskStatus


# ----------------------------------------------------------------- 全文搜索
class SearchMatch(BaseModel):
    novel: str
    chunk_id: int
    chapter_title: str | None = None
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
class RetrievalCandidate(BaseModel):
    novel: str
    chunk_id: int
    chapter_title: str | None = None
    rank: int
    score: float | None = None
    score_label: str | None = None
    previous_rank: int | None = None
    selected: bool = False


class AgentStep(BaseModel):
    step: int
    reason: str
    tool: str
    args: dict = Field(default_factory=dict)
    observation: str
    source_ids: list[str] = Field(default_factory=list)
    # M3.5-③：同一次 Agent 运行的所有步骤共享一个 run_id（轻量串联字段，
    # 由 /api/agent/ask 在入口生成后注入；历史记录里的旧步骤没有，保持 None）。
    run_id: str | None = None


class TraceStep(BaseModel):
    step: str
    detail: str
    # 本阶段耗时（毫秒）。可选：历史会话里存的旧记录没有这个字段。
    ms: int | None = None
    stage_key: str | None = None
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    # --- M3.4 查询扩展步骤专用的结构化字段（其余步骤保持默认值）---
    # stage="expand" 标记这是低置信度补救步骤；reasons 记录触发了哪些信号；
    # variants 是生成的改写变体原文；still_no_evidence 表示补救后信号仍然低。
    stage: str | None = None
    reasons: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    still_no_evidence: bool | None = None
    # --- M3.4 整章扩展步骤专用的结构化字段（其余步骤保持默认值）---
    # expansion_mode="chapter" 标记这是整章扩展实验档；evidence_tokens 是拼入
    # prompt 的证据总 token（真实 embedding tokenizer 口径，闸门不可用时为 null）；
    # truncated / truncation_reason 说明是否触发预算截断以及原因。
    expansion_mode: str | None = None
    evidence_tokens: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


class SourceItem(BaseModel):
    novel: str
    chunk_id: int
    # 兼容旧会话和旧索引：升级后未重建时章节名为 null。
    chapter_title: str | None = None
    text: str


# ----------------------------------------------------------------- 会话历史
class StoredTurn(BaseModel):
    turn_index: int
    role: str
    content: str
    sources: list[SourceItem] | None = None
    trace: list[TraceStep] | None = None
    # 只有 Agent Lab 那条链路的对话会有这个字段；普通问答模式恒为 None。
    agent_steps: list[AgentStep] | None = None
    # M3.5-④：本轮问答使用的在线配置快照（模型、路由、prompt 版本等）。
    # 只在带 run_config 落库的 assistant 轮次上有值；旧记录和 user 轮次恒为 None。
    run_config: dict | None = None
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


# ----------------------------------------------------------------- 关系边审核（M4）
# 审核界面的数据契约。边的主键是 (novel, person_a, person_b, relation) 四元组，
# 前端提交审核时原样带回，后端按四元组定位唯一一条边。
class GraphEdgeItem(BaseModel):
    novel: str
    person_a: str
    person_b: str
    relation: str
    weight: int
    # 关系方向，如 "沈砚秋→小顺"；共现边方向未知，为 null
    direction: str | None = None
    confidence: float | None = None
    # explicit = 明确关系陈述；co_occurrence = 仅同段共现
    evidence_type: str | None = None
    source_chunk_ids: list[int] = Field(default_factory=list)
    review_status: str
    # 第一个来源片段原文的前 80 字，帮审核员快速判断；片段已不存在时为 null
    evidence_excerpt: str | None = None


class GraphEdgeList(BaseModel):
    total: int
    limit: int
    offset: int
    edges: list[GraphEdgeItem]


class GraphReviewRequest(BaseModel):
    novel: str
    person_a: str
    person_b: str
    relation: str
    # 只允许通过/拒绝两种动作；把边恢复成 pending 暂不开放（误操作可再改一次）
    status: str


class GraphReviewResult(BaseModel):
    review_status: str


# ----------------------------------------------------------------- 健康检查
class HealthStatus(BaseModel):
    ok: bool
    ready: bool
