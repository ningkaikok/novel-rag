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

（数据表见下节，跑完后填写。）

## 结果

TODO

## 分析：partial 盲区是否突破

TODO

## 是否达到第一档门槛

TODO

## 下一步建议

TODO

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
