# 通用知识库领域边界迁移说明

## 当前状态：Phase 1

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

## 明确不在本阶段

- 不改 `novel_chunks` 或其他数据库 schema
- 不迁移或删除现有数据，不启用双写
- 不实现 Markdown/PDF parser、V2 repository 或通用文档 API
- 不切换前端“书架”界面，不改变小说问答、引用和 Agent 行为

## 后续接入顺序

1. 建立 V2 schema 和可回滚 migration，在 shadow read 中校验片段/引用一致性。
2. 将现有 TXT 流程接入 parser/repository 协议，再增加 Markdown 与文本型 PDF。
3. 让检索器实际消费 `RetrievalScope`，并把 `SourceRef` 接入 API 和前端引用卡。
4. 最后切换通用文档管理界面；稳定后再考虑停用旧 `/api/books` 兼容入口。

首次切换不删除旧表，不引入独立向量数据库、消息队列或多租户权限系统。
