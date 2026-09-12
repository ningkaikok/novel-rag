"""通用知识库领域模型与协议（Phase 1）。

这些类型描述的是文档、版本、片段、引用和检索范围，不代表数据库 schema。
Phase 1 只建立稳定的领域边界；当前 ``novel_chunks`` 仍由小说适配器负责读写。

兼容约定：

* ``schema_version`` 标识序列化形状，而不是数据库迁移版本；形状变化时递增。
* ``DocumentVersion.version_no`` 从 1 开始，``version_id`` 必须能稳定定位一次
  文档版本；同一版本的引用不能只依赖显示标题。
* ``SourceLocator`` 的 ``kind`` + ``value`` 是机器定位主键，``label`` 只是展示文本。
* ``SourceRef`` 可以安全地跨 API、trace 和评测传递；正文只留在 ``DocumentChunk.text``
  或受版权限制的短 ``excerpt`` 中。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOMAIN_SCHEMA_VERSION: Literal["1"] = "1"
"""通用领域对象的序列化版本；与数据库 migration 版本刻意分离。"""

SourceType = Literal["novel", "text", "markdown", "pdf", "web", "other"]
LocatorKind = Literal["chunk", "page", "heading", "offset"]


class DomainModel(BaseModel):
    """所有领域对象共享的不可变、禁止额外字段的基础配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1"] = DOMAIN_SCHEMA_VERSION


class Collection(DomainModel):
    """一组可被共同检索的文档。"""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""


class Document(DomainModel):
    """文档的稳定身份；内容变化由 ``DocumentVersion`` 表达。"""

    id: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_type: SourceType
    metadata: dict[str, object] = Field(default_factory=dict)


class DocumentVersion(DomainModel):
    """文档内容的一次可复现版本。"""

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    version_no: int = Field(ge=1)
    source_hash: str | None = None
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)


class DocumentChunk(DomainModel):
    """可被检索的文档片段；``ordinal`` 在同一版本内从 0 开始且保持稳定。"""

    id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    context: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class SourceLocator(DomainModel):
    """引用的机器定位信息；展示层不应把 ``label`` 当作定位主键。"""

    kind: LocatorKind
    value: str = Field(min_length=1)
    label: str | None = None


class SourceRef(DomainModel):
    """不携带完整正文的可核验来源引用。"""

    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    source_type: SourceType
    version_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    locator: SourceLocator
    excerpt: str = Field(default="", max_length=80)


class RetrievalScope(DomainModel):
    """检索范围的通用表达；空字段表示不限制该层级。

    允许只指定 ``document_id``，以便未来直接查询文档；指定版本时必须同时指定
    文档，避免出现无法解释的孤立 ``version_id``。
    """

    collection_id: str | None = None
    document_id: str | None = None
    version_id: str | None = None

    @model_validator(mode="after")
    def validate_hierarchy(self) -> RetrievalScope:
        if self.version_id and not self.document_id:
            raise ValueError("version_id 必须与 document_id 一起使用")
        return self

    def cache_fingerprint(self) -> str:
        """返回稳定、与显示文本无关的范围指纹，供查询缓存键使用。"""

        raw = "|".join(
            value or "-" for value in (self.collection_id, self.document_id, self.version_id)
        )
        return sha256(raw.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class DocumentParser(Protocol):
    """未来 TXT/Markdown/PDF parser 的最小协议；Phase 1 只定义，不实现。"""

    name: str
    version: str

    def parse(self, payload: bytes, *, title: str) -> Sequence[DocumentChunk]: ...


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """通用检索器协议；旧 NovelRAG 可通过 LegacyNovelAdapter 接入。"""

    def retrieve(
        self,
        question: str,
        *,
        scope: RetrievalScope | None = None,
        top_k: int = 5,
    ) -> Sequence[DocumentChunk]: ...


@runtime_checkable
class DocumentRepository(Protocol):
    """未来文档仓储的最小读接口，避免领域层依赖 PostgreSQL。"""

    def get_document(self, document_id: str) -> Document | None: ...

    def list_documents(self, *, scope: RetrievalScope | None = None) -> Sequence[Document]: ...


def model_payload(model: DomainModel) -> Mapping[str, object]:
    """以 JSON-safe mapping 导出领域对象，供 trace/缓存等边界使用。"""

    return model.model_dump(mode="json")
