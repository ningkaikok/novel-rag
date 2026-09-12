# 通用知识库领域边界迁移说明

## 当前状态：Phase 3（解析器与索引入口基础已完成）

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

V2 使用 `STORAGE_SCHEMA=v1|v2|shadow` 预留开关，默认值为 `v1`。当前代码不会因为该配置
自动把 NovelRAG/API 切到 V2；切换仍需后续阶段实现并经过 shadow 对比。

## 明确不在本阶段

- 不改 `novel_chunks` 或其他 V1 数据库表；V2 DDL 仅定义在独立 schema 中，默认不执行
- 不迁移或删除现有数据，不启用双写，不切换生产读写
- 本阶段虽已实现 Markdown/PDF parser，但尚未把 parser 接入 V2 repository/数据库索引入口；
  仍不执行 V1→V2 数据写入
- 不切换前端“书架”界面，不改变小说问答、引用和 Agent 行为

## 后续接入顺序

1. 在备份副本或临时数据库显式执行 V2 DDL，并实现 plan 的事务内 upsert。
2. 在 shadow read 中校验片段/引用/检索结果一致性，保留 V1 回滚开关。
3. 将 parser 接入 V2 repository/索引发布入口，再增加 parser 级质量评测。
4. 让检索器实际消费 `RetrievalScope`，并把 `SourceRef` 接入 API 和前端引用卡。
5. 最后切换通用文档管理界面；稳定后再考虑停用旧 `/api/books` 兼容入口。

首次切换不删除旧表，不引入独立向量数据库、消息队列或多租户权限系统。
