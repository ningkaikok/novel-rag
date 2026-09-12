# 通用知识库领域边界迁移说明

## 当前状态：Phase 5A（检索/引用通用化后端最小切片已完成）

本阶段的目标是让小说 RAG 有一个可复用的通用领域语言，同时保持旧系统可运行：

```text
Collection → Document → DocumentVersion → DocumentChunk → SourceRef
                                      ↑
                              RetrievalScope
```

实现位于：

- `src/domain_models.py`：领域模型和 `DocumentParser`、`KnowledgeRetriever`、
  `DocumentRepository` 协议。`schema_version` 是序列化形状版本，与数据库迁移版本分离。
- `src/legacy_novel.py`：唯一处理小说专有字段的适配器。旧小说名生成确定性的
  collection/document/version ID；`chapter_title` 映射为 `section_path`，旧 `chunk_id`
  映射为同一版本内的 `ordinal` 和 `chunk` locator。
- `src/chunk_model.py`：现有 `SourceChunk` 通过适配器提供 `to_document_chunk()` 和
  `to_source_ref()`，旧字段和检索排序完全保留。
- `backend/query_cache.py`：`CacheKey.scope_fingerprint` 为可选字段，旧调用不需要修改；
  未来 collection/document/version scope 会进入缓存隔离。
- `src/v2_schema.py`：在独立的 `knowledge_v2` schema 中定义幂等 DDL；显式
  `apply_v2_schema(conn, dimension)` 才会执行，应用启动不会自动调用。
- `src/v2_migration.py`：把 V1 snapshot 转为确定性的 V2 migration plan，并校验父子关系、
  重复键、chunk 定位、term 外键和 manifest/hash 一致性。
- `scripts/migrate_legacy_to_v2.py`：默认只读取 JSON snapshot 并输出 dry-run 摘要；没有
  数据写入或生产切换选项。
- `src/parsers.py`：提供 `TxtParser`、`MarkdownParser` 和 `PdfTextParser`。TXT 复用现有
  小说清洗/章节/切分规则；Markdown 保留 heading `section_path`；PDF 优先 pdfplumber、
  回退 pypdf，只接受文本型 PDF 并保留 `page_number`/页 locator。
- `ParserLimits` 在进入 parser 前限制字节数、PDF 页数和输出片段数；parser name/version、
  source hash 与切分配置写入 `DocumentVersion`/chunk metadata，供后续索引指纹使用。
- `src/v2_repository.py`：接收已由现有 embedding/BM25 路径计算好的产物，先校验 embedding
  维度、source/pipeline hash、连续 chunk ordinal 和 term 关系，再生成确定性的事务 upsert
  顺序。同一 version 重发布时先按版本删除旧 `chunk_terms`/`document_chunks`，再写入
  当前完整集合，避免缩减 chunk 后残留；collection、document、version、chunk、term
  写入完成后才写 `index_manifests`。
  repository 不加载模型、不重复分词、不自动创建 schema。
- `src/v2_shadow.py`：提供 V1/V2 快照比较纯函数，按文档/版本身份、source hash、chunk
  数量、chunk 稳定键、机器 locator 和检索候选稳定键分类 mismatch；不比较浮点向量，
  不宣称向量召回完全一致。
- `src/retrieval_scope.py`：提供 `RetrievalScope` 到 V1 小说名单的安全投影；未传 scope
  返回兼容的未限制状态，无法证明映射关系时返回空集合，不会扩大为全库。
- `src/retrieval_mixins.py` / `src/rag.py`：V1 向量、BM25、结构性和 hybrid 检索可接收
  可选 scope；collection/document 映射到 `only_novels`，version 仅在 V1 manifest
  提供匹配 `source_hash` 时生效，仍复用参数化 SQL。
- `src/response_adapters.py`：提供通用 `DocumentChunk`、旧 `SourceChunk` 到
  `SourceRef`/JSON-safe payload 的纯适配，保留 document/version/chunk/locator 身份链，
  `excerpt` 限制为 80 字以内。

V2 使用 `STORAGE_SCHEMA=v1|v2|shadow` 预留开关，默认值为 `v1`。当前代码不会因为该配置
自动把 NovelRAG/API 切到 V2；Phase 5A 的 scope 只投影到现有 V1 检索，V2 read/shadow
仍需后续阶段实现并经过真实数据验证。

## Phase 4 的安全边界

- 不改 `novel_chunks` 或其他 V1 数据库表；V2 DDL 仅定义在独立 schema 中，默认不执行
- 不连接真实 PostgreSQL；本阶段只提供 mock executor 可验证的 repository contract
- 不自动执行 apply：`dry_run_v2_index` 只返回摘要；`publish_v2_index` 必须由调用者显式
  选择 `STORAGE_SCHEMA=v2` 或 `shadow`，默认 `v1` 会拒绝发布
- apply 不创建 schema、不删除数据、不切生产读写；调用者需先显式执行已有的
  `apply_v2_schema`，并自行控制数据库连接权限与事务生命周期
- 不切换前端“书架”界面，不改变小说问答、引用和 Agent 行为

## 回滚边界

- V1 是默认且唯一的在线读写路径；V2 发布失败时事务 executor 必须 rollback，manifest
  是最后写入对象，未发布的版本不会被视为可检索索引。
- V2 重发布采用版本内原子 replace，而不是只依赖 `ON CONFLICT`：旧 chunk/term 的清理和
  新索引写入在同一事务内完成；如果中途失败，rollback 会恢复清理前的 V2 状态。
- 即使 V2 已在独立 schema 中发布，回滚只需保持 `STORAGE_SCHEMA=v1`，不读取 V2；V1
  表和数据不会被删除或覆盖。
- 清理 V2 独立 schema、表或数据尚未提供自动化操作，未来必须作为单独、显式、经备份
  确认的运维动作执行；本阶段不会隐式 DROP 或迁移。
- Phase 5A 的 scope 失败安全规则是：未知 collection/document/version 返回空结果；不
  将未知范围降级为全库。回滚仍保持 `STORAGE_SCHEMA=v1`，不需要改变 V1 表或数据。

## 后续接入顺序

1. 在备份副本或临时数据库显式执行 V2 DDL，并把 parser 输出、现有 embedding/BM25
   结果接入 `V2IndexInput`；当前仅完成发布契约，尚未执行真实数据库写入。
2. 在 V2 read/shadow 中接入真实 V1/V2 检索候选，积累 mismatch 观测和 parser 级检索评测；
   当前只有纯函数和 V1 投影，尚未连接真实 V2 数据库。
3. 将 `SourceRef` adapter 接入 API/trace response，并让 V2 retriever 实际消费
   `RetrievalScope`；Phase 5A 尚未修改 API schema。
4. 最后切换通用文档管理界面；稳定后再考虑停用旧 `/api/books` 兼容入口。

首次切换不删除旧表，不引入独立向量数据库、消息队列或多租户权限系统。
