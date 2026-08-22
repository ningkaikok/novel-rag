# M3.4 检索实验：chunk 粒度与 embedding 模型对照（第一轮）

- 日期：2026-08-22
- 语料：`tests/ci_corpus/`（两篇原创短篇，约 1600 字/篇；**小语料，结论不外推到大部头**）
- 工具：`scripts/eval_matrix.py`（隔离临时库，正式索引全程只读）
- 原始数据：`m34_chunk_matrix.json`、`m34_bgem3_matrix.json`（同目录）

## 结果矩阵

| 配置 | 模型 | 片段 | recall@1 | recall@3 | MRR | 路由 | 平均延迟 | 存储 |
|---|---|---|---|---|---|---|---|---|
| chunk500（基线） | bge-small-zh | 6 | 0.800 | 1.000 | 0.900 | 1.000 | 68ms | 192KB |
| chunk300 | bge-small-zh | 11 | **0.600** | 1.000 | 0.783 | 1.000 | 77ms | 240KB |
| chunk800 | bge-small-zh | — | ❌ 被索引质量门禁拦截（>512 token） | | | | | |
| m3_chunk500 | bge-m3 | 6 | 0.800 | 1.000 | 0.900 | 1.000 | 180ms | 272KB |
| m3_chunk800 | bge-m3 | 4 | 0.800 | 1.000 | 0.900 | 1.000 | 99ms | 208KB |

## 发现

1. **chunk300 变差了**。片段更碎让候选池变大、粒度变细，但 top-1 命中率从 0.8 掉到
   0.6——答案被切到相邻片段后，最相关的那个片段不再包含完整关键词上下文。
   recall@3 不变说明"找得到"，是排序质量问题。在当前语料上 **chunk=500 的默认值
   得到验证**，不需要调。
2. **chunk800 被 M3.3 门禁正确拦截**。bge-small-zh 最大序列 512，500 字符的 chunk
   加 overlap 后已接近极限；这不是门禁误伤，而是"该配置在本模型下根本不可行"——
   过去这类配置会静默截断、指标莫名下降还查不到原因。
3. **BGE-M3 dense 在本语料无质量收益，成本全面更高**：索引耗时 ×2.2、平均查询
   延迟 ×2.6、向量存储 ×1.4（1024 维 vs 512 维）。它的真正差异化能力（sparse /
   multi-vector）需要额外检索通路才能发挥，属于递进矩阵的第③④步。
4. **BGE-M3 打开了 chunk800 的可行性**（8192 token 上下文），且 chunk800 用更少
   片段拿到相同指标。若未来换用长上下文 embedding 模型，大 chunk 值得重测。

## 下一步

- 在真实大部头语料（本地库，只读评测）上复跑本矩阵，验证结论是否迁移；
- BGE-M3 第③步（dense+sparse，需 pgvector `sparsevec` 列与稀疏检索通路）、
  第④步（multi-vector late interaction）需要新的 schema 支持，单独立项；
- 低置信度阈值用 `scripts/eval_low_confidence.py` 在带失败案例的真实评测集上校准。

## 复现

```bash
python scripts/eval_matrix.py --output docs/experiments/m34_chunk_matrix.json
python scripts/eval_matrix.py \
  --config '{"name":"m3_dense_chunk500","env":{"EMBEDDING_MODEL":"BAAI/bge-m3"}}' \
  --config '{"name":"m3_dense_chunk800","env":{"EMBEDDING_MODEL":"BAAI/bge-m3","CHUNK_SIZE":"800","CHUNK_OVERLAP":"120"}}'
```
