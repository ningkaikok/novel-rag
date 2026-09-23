# M3.5 两步走 Judge 实验：先抽断言再逐条核对能否突破「半真半假」盲区

- 日期：2026-08-25
- 背景：第二阶段校准（[m35-faithfulness-calibration.md](m35-faithfulness-calibration.md)）
  的结论是所有单步 Judge 都退化成二分类器，对「半真半假」系统性失明——
  标注集里 26% 是 partial，而三个 Judge 几乎从不输出 partial，一致率天花板被钉死在
  62%~68%。第三阶段按其「下一步」第 1、2 条实现并复测**两步走 prompt**：
  第一步让 Judge 从陈述中抽取可独立核对的原子断言（结构化输出），第二步逐条断言对照
  检索片段判定 supported/contradicted/not_found，最后用机械规则聚合出最终标签——
  partial 从"模型愿不愿意说"变成聚合规则的必然产物。
- 标注集：`tests/citation_shadow_set.json`（53 条；supported 22 / partial 14 /
  unsupported 17），与第二阶段完全相同，数字可直接对照
- 工具：`scripts/eval_faithfulness_shadow.py`（新增 `two:` 方法前缀 / `--two-step` 开关 /
  `FAITHFULNESS_JUDGE_MODE` 环境变量；`src/citation_eval.py` 新增
  `extract_claims` / `judge_claim` / `aggregate_claim_verdicts` / `judge_support_two_step`，
  与单步 `judge_support` 并存，默认行为不变）
- 成本控制：每档只跑一轮，不做自一致性投票（后续项）；glm-4-flash 免费档，
  air 为低价档，sonnet 走本机 claude CLI 订阅额度

本轮实际评测日期：2026-09-23。受当前环境可用模型限制，先使用本机 Ollama
`qwen2.5:7b` 对同一 53 条标注集做单步与两步对照，不调用云端模型。
两步版用批量 verdict 模式（断言抽取一次、全部断言核对一次），避免逐断言调用造成
长时间评测；与历史 GLM/Claude 结果只作方向性参考。
主表的单步与两步运行都使用未加 Ollama JSON 解码约束的初版路由；之后增加约束并对失败
样本做的定向复跑只用于诊断，不回写主表指标。

## 结果

| 方法 | 一致率 | 支持 P | 支持 R | 反对 P | 反对 R | uncertain | 调用失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 规则基线 | 5.7% | 75% | 14% | — | 0% | 92.5% | 0 |
| Ollama Qwen2.5 7B 单步 | 64.2% | 63% | 100% | 67% | 71% | 0.0% | 0 |
| Ollama Qwen2.5 7B 两步（批量核对） | 54.7% | 84% | 73% | 71% | 29% | 5.7% | 3 |

逐人工类别一致数：单步 supported **22/22**、partial **0/14**、unsupported **12/17**；
两步 supported **16/22**、partial **8/14**、unsupported **5/17**。两步结果的 3 条
调用失败在重试后保留为 uncertain，并计入总分母。

逐条输出与两步断言分别保存于
[`m35-qwen25-7b.json`](m35-qwen25-7b.json) 和
[`m35-qwen25-7b-two-step-batch.json`](m35-qwen25-7b-two-step-batch.json)。

## 分析：partial 盲区是否突破

两步 Judge 把 partial 命中从 **0/14 提高到 8/14（+57.1 个百分点）**，说明“先拆断言、
再机械聚合”确实缓解了单步模型完全不输出 partial 的问题。但代价是整体一致率从
64.2% 降到 54.7%，unsupported 一致数从 12/17 降到 5/17；模型会把部分有据但另一部分
无据/矛盾的样本判为 partial，这对人工 partial 有帮助，也把若干人工 unsupported
错误地抬成 partial。两步法改变了错误分布，尚未带来整体质量提升。

本轮只有本机 Qwen2.5 7B，且第二步采用批量核对，因此这是本地可复现实验，不代表
跨模型结论。3 条 JSON 解析失败也说明结构化输出稳定性仍需改进；结果文件保留了
降级状态，不能把失败样本当作模型判断正确。

## 是否达到第一档门槛

**未达到第一档门槛，继续保持影子模式。** 两步版本一致率 54.7%（门槛 ≥80%）、支持
精确率 84%（门槛 ≥95%）、反对召回率 29%（门槛 ≥90%），三项均未通过。单步同样未通过。
不得接入问答主链路、UI 提示或拒答策略。

## 下一步建议

1. 在报告中记录了本地单步与批量两步的完整结果，可先据此判断方向：两步 prompt
   能产出 partial，但目前是以总体准确率和 unsupported 召回下降为代价。
2. 评测路由现对 Ollama 请求 JSON 输出格式和 1,024 token 生成预算。对本轮 3 条解析
   失败样本的定向复跑中，1 条恢复为 partial，另 2 条仍漏掉 verdict index=1；下一步应
   改善批量输出的结构约束或补齐策略，再重跑完整集并保留原始失败记录。
3. 若继续推进，扩充 partial 与 unsupported 的困难样本，并额外跑逐断言核对模式，分离
   prompt 拆解收益与批量判断误差。仍以第一档三项门槛共同通过为准。
4. 当前结果不足以批准 UI 或线上策略变更；复测前继续仅作为离线实验。

## 运行环境事故记录：代理僵尸流与两级看门狗

第一轮全量评测（9 方法）耗时异常且中途挂死 25 分钟，定位过程与结论值得留档：

**现象**。`two:glm:glm-4.5-air` 开跑后日志冻结 25 分钟无输出，进程 CPU <1%；
`sample` 采样显示主线程阻塞在 `_ssl__SSLSocket_read → PySSL_select → poll`，
即等一个永远不会来的 SSE 数据块。

**根因链**。
1. 本机代理（Surge/Clash fake-ip 模式，连接特征 `198.18.x.x -> :443`）的隧道
   间歇性卡死，但 TLS 层仍间歇喂入心跳字节——requests 的读超时 `(10, 300)`
   只约束**相邻数据块的间隔**，每次心跳都把 300s 计时器重置，于是流被挂成
   「永远读不完也永不超时」的僵尸连接；
2. 第一版修复尝试 SIGALRM 墙钟看门狗：隔离测试对 `time.sleep` 完美生效
   （到点抛异常），但对真实 SSL read **无效**——macOS 上 `_ssl` 的 C 层
   select/poll 循环在 EINTR 后自行重试，不返回 Python 解释器边界，
   信号处理器里 raise 的异常永远没机会浮出。

**最终修复**（两层，各管一段）：
- `backend/zhipu.py`：新增 `ZHIPU_STREAM_DEADLINE` 环境变量——后台线程
  `threading.Timer` 到点强制 `resp.close()`，从另一线程关套接字让阻塞中的
  read 立刻抛错。这是唯一不依赖信号语义的可靠打断方式；默认关闭，不影响
  其他调用方（问答主链路有自己的超时策略）。
- `scripts/eval_faithfulness_shadow.py`：进度打印全部加 `flush=True`
  （此前 stdout 块缓冲导致后台运行时无法判断进程是慢是死），claude CLI
  路径保留 SIGALRM 看门狗（子进程等待路径信号可打断）。

看门狗触发后异常落在既有可重试语义上（「Judge 调用失败：」前缀），
`judge_with_retry` 零改动即可退避重试。

**成本核算**。两步走的调用次数是单步的 4~7 倍（抽断言 1 次 + 逐条核对
2~6 次）；叠加 GLM 免费档/低价档与 claude CLI 订阅额度的流式延迟
（单次 10~60s）、0.5s 全局限速闸，9 个方法 × 53 例 ≈ 1500+ 次调用，
纯算力时间约 2~3 小时；第一轮因僵尸流挂死报废重跑，实际墙钟翻倍。
后续扩大标注集前先把 `--dump-json` 改成**增量落盘**（当前只在结束时写），
否则再遇中断只能从头烧钱——已列入下一步建议。

## 复现

```bash
uv run python scripts/eval_faithfulness_shadow.py \
  --model glm:glm-4-flash --model two:glm:glm-4-flash \
  --model two:glm:glm-4.5-air \
  --model claude:sonnet --model two:claude:sonnet
```
