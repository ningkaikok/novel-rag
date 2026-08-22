# M3.4 检索实验：chunk 粒度与 embedding 模型对照（第一轮）

- 日期：2026-08-22
- 第一轮语料：`tests/ci_corpus/`（两篇原创短篇，约 1600 字/篇）
- 第二轮（大部头验证）：《凡人修仙传》全本（15MB，19901 片段），8 条专用评测用例
- 工具：`scripts/eval_matrix.py`（隔离临时库，正式索引全程只读；本轮新增
  `--corpus-dir/--test-set` 参数支持受控大部头样本）
- 原始数据：`m34_chunk_matrix.json`、`m34_bgem3_matrix.json`、`m34_bigbook_matrix.json`

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

- ~~在真实大部头语料上复跑本矩阵~~（已完成，见下文第二轮）；
- BGE-M3 第③步（dense+sparse，需 pgvector `sparsevec` 列与稀疏检索通路）、
  第④步（multi-vector late interaction）需要新的 schema 支持，单独立项；
- 低置信度阈值用 `scripts/eval_low_confidence.py` 在带失败案例的真实评测集上校准；
- 排查大部头秒级检索延迟的阶段占比（复用 M3.1 trace 的分阶段耗时）。

## 大部头验证：《凡人修仙传》全本（第二轮）

| 配置 | 片段 | 索引耗时 | 存储 | recall@1 | recall@3 | MRR | 平均延迟 |
|---|---|---|---|---|---|---|---|
| c500 基线（bge-small） | 19,901 | 261s | 190MB | **0.750** | 0.750 | 0.768 | 2262ms |
| chunk300（bge-small） | 35,108 | 347s | 316MB | **0.625** ↓ | 0.750 | 0.688 ↓ | 2602ms |
| BGE-M3 dense c500 | 19,901 | **2907s** ↑↑ | 450MB | 0.750 | 0.750 | 0.771 | 3317ms |

**小语料的三个结论全部迁移，且大部头上更显著：**

1. **chunk300 结论迁移成立**：recall@1 从 0.750 掉到 0.625（-12.5pp），MRR 从
   0.768 掉到 0.688。长篇小说情节跨度大、指代更多，碎片化对 top-1 的伤害比短篇
   更重。chunk=500 的默认值在两类语料上都得到验证。
2. **BGE-M3 dense 无收益的结论强化**：质量与基线完全持平（recall@1 相同、
   MRR +0.003 在噪声范围内），但索引耗时 ×11（261s → 2907s，约 48 分钟）、
   单次查询延迟 +47%、存储 ×2.4。纯 dense 场景下换它只有成本没有回报——它的
   价值必须靠 sparse / multi-vector 检索通路兑现（递进第③④步）。
3. **新发现：大部头的绝对延迟已到秒级**（基线单次混合检索 2.3s，BGE-M3 3.3s）。
   延迟随语料规模明显增长，后续值得排查瓶颈在 embedding 编码还是 HNSW/RRF/
   重排各阶段的占比——这正好可以复用 M3.1 的检索 trace 分阶段耗时数据。

实验成本备注：BGE-M3 在大部头上一轮索引约 48 分钟；递进第③④步若在大部头
规模上做对照，需要预算数小时级索引时间或改用采样切片。

## 复现

```bash
python scripts/eval_matrix.py --output docs/experiments/m34_chunk_matrix.json
python scripts/eval_matrix.py \
  --config '{"name":"m3_dense_chunk500","env":{"EMBEDDING_MODEL":"BAAI/bge-m3"}}' \
  --config '{"name":"m3_dense_chunk800","env":{"EMBEDDING_MODEL":"BAAI/bge-m3","CHUNK_SIZE":"800","CHUNK_OVERLAP":"120"}}'
```
