# Agent Lab：用普通 Python 看懂工具循环

Agent Lab 是标准 RAG 旁边的独立实验模式。标准 RAG 的路线固定、稳定，适合日常问答；
Agent Lab 让模型根据观察选择下一步，适合学习 Agent 的状态和控制流。

```text
问题 + 已有观察
  → 模型只输出一个 JSON action
  → Python 校验工具白名单与参数
  → 执行只读工具，登记证据 S1/S2/...
  → 同时登记结构化事实与覆盖范围（complete / bounded / partial）
  → 把新观察放回状态
  → 最多 3～5 步，最后依据选中的原文生成引用答案
```

## 工具与能力边界

| 工具 | 用途 |
| --- | --- |
| `query_library` | 统一查询书籍、章节、片段统计；使用受限的 `domain + operation` 参数 |
| `list_books` | 兼容旧 action，内部转发到 `query_library(domain=books, operation=list)` |
| `search_novels` | 用现有混合 RAG 搜索一本或全部小说 |
| `read_neighbors` | 片段被切断时读取前后文，半径最多 3 |
| `get_chapter` | 已知章节时按章节标题读取原文，限制返回数量 |
| `answer_with_citations` | 终止动作，只能使用本轮已经登记的证据 |

前端会逐步显示“选择理由 → 工具 → 观察 → 新证据 ID”。最终答案仍使用普通问答的
`[1]` 引用和原文卡片；S1/S2 是 Agent 状态里的稳定 ID，用来解释它选择了哪些证据。

## 为什么不能只看召回片段

工具结果分成两部分：`summary` 只给规划器快速阅读，`facts` 保存可以校验的结构化事实。
每条事实都带覆盖范围：

- `complete`：可以回答总数、全集列表等范围问题；例如 `list_books` 的书架目录。
- `bounded`：只覆盖明确边界内的数据；例如片段的前后邻居、限制条数的章节读取。
- `partial`：混合检索的 top-k 结果，只说明“召回到了什么”，不说明“数据库里全部有什么”。

因此，“现在一共有几部小说”会先走 `query_library`，再使用完整目录事实；不会把搜索命中的
两本书误认为书架总数。这个机制是覆盖范围门禁，不是某个问题的字符串特判。以后增加
人物目录、章节目录、关系图等工具时，只需让工具返回相应的 `kind` 和 `coverage`，并在
答案策略中注册该事实类型即可复用。

对于要求“全部/所有/一共/有哪些”的其他问题，如果当前工具没有提供 `coverage=complete`
的事实，回答模型必须明确说明证据不完整，而不能用局部片段拼出一个看似确定的全集答案。

当前已经把结构化查询收敛为一个能力型工具 `query_library`：

```text
有哪些章节       → domain=chapters, operation=list
某章有多少片段   → domain=chunks, operation=count, chapter=...
现在有几部小说   → domain=books, operation=list
```

它只接受白名单参数并由 Python 编译固定 SQL，不允许模型传入任意 SQL。后续的人物出现
位置、跨书人物交集属于另一种数据能力，再增加 `query_entities`，而不是为每一种问法增加
一个工具。

## 安全边界

- 工具名是显式白名单，模型不能构造 SQL、运行 shell 或调用未注册函数。
- 所有数据库工具只读，并限制半径/返回条数。
- 没有证据时禁止 `answer_with_citations`，连续失败后明确拒答，不让生成模型凭空补齐。
- 重复动作会被检测；有证据就结束，没有证据则退回原问题搜索。
- 步数在 API 层和循环层双重限制为 3～5，避免失控循环和不可预测费用。

Agent Lab 当前不保存到普通会话历史。这让实验轨迹和稳定聊天记录保持边界，也避免
刷新后把一半执行误当成可恢复工作流。

## 动作解析的可观测性（M3.2.1 前置埋点）

规划器的输出是一段 JSON，`_parse_action` 容忍模型带 Markdown 围栏或在 JSON 前后
多写几句话，但**容忍本身有风险**：正则兜底抓的是第一个 `{...}`，模型多写一句
带花括号的话就可能抓错对象，而且抓错了不会报错。

每个由规划器产出的步骤都带一个 `parse_mode`：`strict`（裸 JSON 一次成功）/
`fenced`（剥掉围栏才成功）/ `regex`（靠正则兜底，最值得关注）/
`failed:<类别>`（彻底失败，走了降级）。它随 `agent_steps` 落库，用

```bash
uv run python scripts/agent_parse_stats.py           # 全部历史
uv run python scripts/agent_parse_stats.py --days 7  # 最近 7 天
```

聚合。这条埋点是路线图 M3.2.1（把 JSON 协议换成首行标签协议）的前置项——先用
真实失败率决定值不值得换，而不是凭"新协议更好"就抢跑。

## 为什么这里仍不需要 LangGraph

这里已经是真正的“工具 → 观察 → 再判断”循环，但状态只有一个 Python 列表，工具只有
五个，最多五步，并且必须在一次 HTTP 请求内完成。普通 `for` 循环更容易单步阅读和测试。

当它需要跨进程检查点、暂停等待人工批准、并行子任务、几十步重试或恢复某个中间节点时，
再把这套已测试的工具函数迁入 LangGraph；框架应解决已经出现的问题，而不是遮住基本原理。

## 从 Agent Lab 到生产架构

Agent Lab 有意把“生产级系统”的复杂边界压缩成几个容易读懂的 Python 结构：工具白名单
对应未来的 Tool Registry，`readonly_toolbox.execute` 对应未来的 Tool Gateway，`S1/S2`
证据表对应未来的状态和事件记录。它是教学起点，不是生产安全边界。

推荐的演进顺序是：

```text
五个函数白名单
  → ToolSpec / Tool Registry
  → Tool Gateway（权限、schema、超时、审计）
  → Model Gateway（能力、预算、超时、显式降级）
  → Agent 事件与评测（run_id、质量、延迟、成本）
  → Chat/Run/Event 状态拆分与长任务恢复
  → 需要跨客户端时再接 MCP
```

几个概念要分开：MCP 是连接工具、资源和提示模板的协议；它不会替代自己的权限和风险
检查。Tool Discovery 解决“给模型看哪些候选工具”，Tool Router 解决“最终选择哪个
工具”，Tool Gateway 才是实际执行前后的统一控制点。完整目标架构和 M6 验收标准见
[Agent 平台化架构](agent-platform-architecture.md)。

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
