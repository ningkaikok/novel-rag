# 检索延迟分阶段画像（M3.4 性能预算排查）

- 日期：2026-08-23
- 对象：《凡人修仙传》真实索引（19,901 片段，本机 PostgreSQL）
- 方法：`scripts/profile_latency.py`，8 个真实评测问题 × 3 轮，从 trace 提取各阶段耗时
- 原始数据：`latency-profile.json`（同目录）

## 结果

| 阶段 | 次数 | P50(ms) | P95(ms) | 均值 | 占比 |
|---|---|---|---|---|---|
| **精排（交叉编码器）** | 24 | 369 | 417 | 484 | **50.8%** |
| **BM25 召回** | 24 | 227 | **602** | 287 | 30.2% |
| 向量召回（HNSW） | 24 | 62 | 281 | 118 | 12.4% |
| 理解问题（查询 embedding） | 24 | 27 | 109 | 58 | 6.1% |
| 结构性召回（按需触发） | 3 | 10 | 14 | 11 | 1.2% |
| **端到端总计** | 24 | 682 | 1118 | 952 | — |

## 发现与归因

### 1. 精排占一半（~480ms）
60 个候选对跑 `bge-reranker-base` 交叉编码器的推理成本。可选优化：
- 候选数 `RERANK_CANDIDATE_MULTIPLIER=3` 调小（质量需过评测门禁验证）
- 批大小/设备调优；或换更小的重排模型对照

### 2. BM25 的 600ms 尾部有明确病灶（EXPLAIN 实锤）
```
Bitmap Index Scan on chunk_terms_term_idx   12ms   ← 索引很快
HashAggregate  rows=15904                  286ms   ← 全量聚合 5 个词的全部命中
Sort (top-N heapsort)                        ~0ms
Execution Time                             286ms
```
常见词（"韩立"）命中数千片段，SQL 先聚合**全部**命中行再排序取 Top-20。
**低成本修复方向**：改为两阶段——SQL 内每词按 tf 取 Top-N，Python 端融合，
预计把 BM25 从 ~290ms 压到 <50ms；语义近似（可能漏掉"单词条频低但多词合计高"
的极端片段），必须过夜间检索门禁验证 Recall 不回退。

### 3. 向量召回健康
HNSW 在 2 万片段上 P50 62ms，无需动。

## 后续动作（已入路线图）

| 项 | 优先级 | 验证方式 |
|---|---|---|
| BM25 两阶段聚合改写 | P2（成本低收益明确） | 夜间评测门禁 + 本画像复测 |
| 重排候选数/批量调优实验 | P2 | eval_matrix 质量×延迟对照 |

## 复现

```bash
uv run python scripts/profile_latency.py --rounds 3
psql ... -c "EXPLAIN ANALYZE <chunk_terms 聚合查询>"   # 见 keyword_retrieve 的 SQL
```
