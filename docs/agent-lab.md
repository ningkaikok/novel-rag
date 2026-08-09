# Agent Lab：用普通 Python 看懂工具循环

Agent Lab 是标准 RAG 旁边的独立实验模式。标准 RAG 的路线固定、稳定，适合日常问答；
Agent Lab 让模型根据观察选择下一步，适合学习 Agent 的状态和控制流。

```text
问题 + 已有观察
  → 模型只输出一个 JSON action
  → Python 校验工具白名单与参数
  → 执行只读工具，登记证据 S1/S2/...
  → 把新观察放回状态
  → 最多 3～5 步，最后依据选中的原文生成引用答案
```

## 五个工具

| 工具 | 用途 |
| --- | --- |
| `list_books` | 不清楚书架范围时列出书名 |
| `search_novels` | 用现有混合 RAG 搜索一本或全部小说 |
| `read_neighbors` | 片段被切断时读取前后文，半径最多 3 |
| `get_chapter` | 已知章节时按章节标题读取原文，限制返回数量 |
| `answer_with_citations` | 终止动作，只能使用本轮已经登记的证据 |

前端会逐步显示“选择理由 → 工具 → 观察 → 新证据 ID”。最终答案仍使用普通问答的
`[1]` 引用和原文卡片；S1/S2 是 Agent 状态里的稳定 ID，用来解释它选择了哪些证据。

## 安全边界

- 工具名是显式白名单，模型不能构造 SQL、运行 shell 或调用未注册函数。
- 所有数据库工具只读，并限制半径/返回条数。
- 没有证据时禁止 `answer_with_citations`，连续失败后明确拒答，不让生成模型凭空补齐。
- 重复动作会被检测；有证据就结束，没有证据则退回原问题搜索。
- 步数在 API 层和循环层双重限制为 3～5，避免失控循环和不可预测费用。

Agent Lab 当前不保存到普通会话历史。这让实验轨迹和稳定聊天记录保持边界，也避免
刷新后把一半执行误当成可恢复工作流。

## 为什么这里仍不需要 LangGraph

这里已经是真正的“工具 → 观察 → 再判断”循环，但状态只有一个 Python 列表，工具只有
五个，最多五步，并且必须在一次 HTTP 请求内完成。普通 `for` 循环更容易单步阅读和测试。

当它需要跨进程检查点、暂停等待人工批准、并行子任务、几十步重试或恢复某个中间节点时，
再把这套已测试的工具函数迁入 LangGraph；框架应解决已经出现的问题，而不是遮住基本原理。

## API 与测试

```http
POST /api/agent/ask
Content-Type: application/json

{"question": "顾长风为什么卧床？", "max_steps": 5}
```

SSE 事件顺序为若干 `agent_step`，然后是 `sources/token/done`。运行：

```bash
python -m pytest tests/backend/test_agent_lab.py
cd frontend && npx playwright test e2e/agent-lab.spec.ts
```
