# M3.4 重排候选数调优实验：RERANK_CANDIDATE_MULTIPLIER 质量×延迟对照

- 日期：2026-08-25
- 背景：延迟画像显示精排（交叉编码器）占端到端检索延迟约 51%（BM25 修复后占比进一步
  升到 ~80%），需要质量×延迟对照数据决定 `RERANK_CANDIDATE_MULTIPLIER` 的最优值
- 对象：《凡人修仙传》真实索引（19,901 片段，全库 33,536 片段，本机 PostgreSQL，全程只读）
- 环境：`bge-reranker-base` CPU 推理；生产口径 TOP_K=3，评测口径 top_k=20
- 工具：`scripts/eval_matrix.py`（隔离临时库）、`scripts/profile_latency.py --rounds 3`、
  `scripts/eval_retrieval.py`
- 原始数据：`m34_rerank_mult_matrix.json`（CI 矩阵）；其余数字由下列命令现场复现

## 先说机制：这个参数在默认配置下是「死」的

候选池公式在 `src/rag.py`：

```
candidate_k = max(top_k × RERANK_CANDIDATE_MULTIPLIER, RECALL_K)
# TOP_K=3，RECALL_K=20
```

| multiplier | top_k×m | 实际 candidate_k |
|---|---|---|
| 2 | 6 | **20**（被 RECALL_K 兜底） |
| 3（默认） | 9 | **20**（被 RECALL_K 兜底） |
| 4 | 12 | **20**（被 RECALL_K 兜底） |
| 6 | 18 | **20**（被 RECALL_K 兜底） |

trace 实证：四档在生产路径的精排步骤 detail 均为「用交叉编码器对 **20** 个候选重新
打分」。也就是说 **TOP_K=3 时 {2,3,4,6} 四档完全等价**——multiplier 要重新生效，
TOP_K 必须 ≥ 7。它真正起作用的场景是评测链路（`eval_retrieval.py` 用 top_k=20），
因此下面分两条路径分别测量。

顺带更正：[延迟画像](m34-latency-profile.md) 里「60 个候选对跑交叉编码器」有误，
实际打分对象是 RRF 截断后的 20 个候选（原文已同步修正）。

## 结果一：生产口径（TOP_K=3，真实库，profile_latency ×3 轮）

| multiplier | 精排 P50/P95(ms) | 精排均值 | 端到端 P50/P95(ms) | 精排占比 |
|---|---|---|---|---|
| 2 | 398 / 430 | 536 | 500 / 1642* | 70.4% |
| 3（默认） | 371 / 411 | 474 | 450 / 572 | 80.2% |
| 4 | 370 / 401 | 462 | 437 / 523 | 83.3% |
| 6 | 416 / 1381* | 753* | 666 / 1654* | 75.8% |

\* 标星行含明显机器噪声（同轮向量召回 P95 也异常抬高）。有效结论：
四档精排 P50 稳定在 370~420ms 区间，**没有任何系统性差异**——与「候选数恒为 20」
的机制判断一致。精排在 BM25 修复后占端到端 ~70~83%，仍是第一瓶颈。

## 结果二：评测口径（真实库 13 条用例，top_k=20，multiplier 在此生效）

| multiplier | 候选池 | recall@1 | recall@3 | MRR | 路由 | 平均耗时 |
|---|---|---|---|---|---|---|
| 2 | 40 | 0.538 | 0.615 | 0.590 | 0.923 | 1152ms |
| 3（默认） | 60 | 0.538 | 0.615 | 0.590 | 0.923 | 1459ms |
| 4 | 80 | **0.462 ↓** | 0.615 | **0.549 ↓** | 0.923 | 1737ms |
| 6 | 120 | **0.462 ↓** | 0.692 ↑ | 0.575 ↓ | 0.923 | 2431ms |

逐用例对照（相对默认 m=3）：

| 对照 | 变化用例 |
|---|---|
| m=2 vs m=3 | 13 条全部一致 |
| m=4 vs m=3 | Q16 #1→#2 ↓、Q14 #6→#7 ↓（无新增命中） |
| m=6 vs m=3 | 同上两条回退 + Q17 未命中→#3 ↑ |

两个发现：

1. **延迟线性**：每 +20 个候选约 +300ms（≈15ms/对），m=6 比 m=2 慢一倍多。
2. **质量非单调，更大不更好**：候选池扩大后，交叉编码器有机会把低 RRF 位次的
   片段救回（Q17 在 120 池才进候选、重排到 #3），但同样会放进更多干扰片段把
   正确片段挤出前排——m=4/6 各损失两个前排命中，MRR 净下降。在本评测集上
   「召回 20 → 重排挑」的池子已经够大，继续加大是净伤害。

## 结果三：CI 小语料（eval_matrix，隔离临时库）

| 配置 | 片段 | recall@1 | recall@3 | MRR | 路由 | 平均延迟 |
|---|---|---|---|---|---|---|
| mult2 / mult3 / mult4 / mult6 | 6 | 全部 0.800 | 全部 1.000 | 全部 0.900 | 全部 1.000 | 69~232ms（噪声内） |

语料只有 6 个片段，任何档位的候选池都全量覆盖，指标逐位相同——**小语料对该参数
没有区分度**，这正是结论必须靠大部头验证的原因。默认档按夜间门禁同口径跑
`--strict --tolerance 0.05` 对照 `tests/ci_eval_baseline.json`：通过，无回退。

## 附：批大小探针（路线图该项的另一半）

120 个真实候选对，CPU 上测 `CrossEncoder.predict(batch_size=b)` ×3 取中位：

| batch_size | 8 | 16 | **32（默认）** | 64 | 128 |
|---|---|---|---|---|---|
| 耗时 P50(ms) | 2005 | 2003 | 2015 | 2111 | 2314 |

批大小不是杠杆：默认 32 与最优差 <1%，再大反而变慢。不改。

## 结论

**保持默认值 `RERANK_CANDIDATE_MULTIPLIER=3` 不变**，理由：

1. 生产路径（TOP_K=3）下 {2,3,4,6} 被 `RECALL_K=20` 兜底为同一候选池，改它没有任何
   效果——这是本实验最重要的机制性发现；
2. multiplier 重新生效的场景（top_k ≥ 7 或未来调大 TOP_K）里，实测安全区间是
   2~3：m=2 与 m=3 全部用例等价且更省，m≥4 出现 recall@1 回退，默认 3 居安全区中点；
3. 生产精排延迟能否再降，取决于候选池下限（`RECALL_K`，需与不开重排的退回路径、
   评测 FETCH_K 一并权衡）或设备/模型层优化，不属于本参数的调节范围，另行立项。

## 复现

```bash
# CI 小语料矩阵 + strict 门禁
uv run python scripts/eval_matrix.py \
  --config '{"name":"mult2","env":{"RERANK_CANDIDATE_MULTIPLIER":"2"}}' \
  --config '{"name":"mult3","env":{"RERANK_CANDIDATE_MULTIPLIER":"3"}}' \
  --config '{"name":"mult4","env":{"RERANK_CANDIDATE_MULTIPLIER":"4"}}' \
  --config '{"name":"mult6","env":{"RERANK_CANDIDATE_MULTIPLIER":"6"}}' \
  --output docs/experiments/m34_rerank_mult_matrix.json

# 生产口径延迟画像（每档一轮，串行避免互相干扰）
for m in 2 3 4 6; do
  RERANK_CANDIDATE_MULTIPLIER=$m uv run python scripts/profile_latency.py --rounds 3
done

# 大部头评测口径（13 条真实库用例）
for m in 2 3 4 6; do
  RERANK_CANDIDATE_MULTIPLIER=$m uv run python scripts/eval_retrieval.py
done
```
