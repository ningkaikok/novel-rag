# M3 层级检索：从片段走向全书问题

普通 RAG 擅长“某句话在哪里”，但主题、人物成长和跨书比较的证据通常散落在几十章。
只从全书片段中取一个很小的 top-k，容易碰巧只看见故事早期或某个局部。M3 为此增加
一层导航索引：

```text
原文片段 novel_chunks
  → 按章节标题聚合（无标题时按固定窗口形成虚拟章节）
  → 章节摘要 hierarchy_summaries(level=chapter)
  → 全书摘要 hierarchy_summaries(level=novel)
```

## 查询时怎样工作

`src/hierarchy.py:is_global_question` 只对“全书、主题、成长、变化、比较”等明确的
全局信号开启层级检索，局部事实问题仍走原来的片段检索，避免平白增加噪声。

1. 没有点名书时，先搜索全书摘要，判断应该进入哪些书。
2. 在每本目标书内分别搜索章节摘要；跨书问题按书分配配额，避免一本书包揽候选。
3. 摘要命中后，用 `start_chunk_id/end_chunk_id` 回到章节开头、中间、结尾的原文。
4. 原文候选和向量、BM25、结构性召回一起进入 RRF，再由交叉编码器重排。
5. 最终 prompt 和 `[n]` 引用只包含 `novel_chunks.text`，摘要永远不冒充原文证据。

这条“不直接依据摘要回答”的边界很重要：摘要会压缩细节，也可能遗漏事实。它适合
导航，不适合成为不可核验的事实来源。

## 摘要为什么暂时不用 LLM

当前 `extract_summary` 采用确定性的“开头 + 均匀中段 + 结尾”抽取策略。优点是离线、
免费、可重复，几千章小说也不会因为云端限流留下一半索引。缺点是它不是文学分析，
对隐含主题的概括能力有限。

这是一条刻意保留的学习边界：先把层级数据模型、检索和证据回溯做对，再用相同评测集
替换摘要器做 A/B。以后接 LLM 时只需替换 `extract_summary`，表结构与查询链路不用重写。

## 数据表与增量迁移

- `hierarchy_summaries`：节点层级、标题、原文范围、摘要和 pgvector embedding。
- `hierarchy_manifest`：每本书的文件哈希、层级算法指纹和节点数。

层级算法拥有独立的 `hierarchy_pipeline_hash`。从 M2 升级时，基础片段索引没变的书只
补建层级节点，不重算 3 万多个片段向量或 BM25。新增/修改书则在同一个单书事务里同时
切换片段索引和层级索引。

```bash
python src/ingest.py
```

运行后可查看 `hierarchy_nodes` 数量。设 `HIERARCHY_ENABLED=0` 可以关闭查询与新建，
已有表不会被删除。

## 怎么验证

纯算法和“摘要命中必须回到原文”的测试：

```bash
python -m pytest tests/backend/test_hierarchy.py tests/backend/test_incremental_ingest.py
```

主题、人物成长、跨书覆盖的专用评测：

```bash
python scripts/eval_hierarchy.py
```

评测关注目标书覆盖、章节多样性和证据位置跨度，不用一个关键词假装能代表“主题分析
是否完整”。生成答案是否合理仍需要人工或模型评审，这是当前指标的明确边界。
