# Agent 平台化架构：从 Agent Lab 走向生产边界

> 状态：**规划中，不代表当前版本已经实现**
>
> 适用范围：把当前五步只读 Agent Lab 演进为可治理、可观测、可恢复的 Agent 后端

这份文档描述的是目标架构和演进顺序，不是要求现在一次性部署 Redis、消息队列、
MCP、LangGraph 和多个微服务。学习项目最重要的是先把边界拆清楚，再按真实需求增加
基础设施。

## 一、优化后的目标架构

```text
                         Control Plane
  ┌──────────────────────────────────────────────────────┐
  │ Tool / Model Registry │ Policy │ Prompt │ Eval │ Rollout │
  └───────────────────────────┬──────────────────────────┘
                              │ 版本化运行快照
                              ▼
                          Data Plane
Client → API Gateway → Agent API / Runtime
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        Domain Router      Planner       Model Gateway
                              │          Ollama/Claude/GLM
                       Tool Discovery
                              │ 权限/状态/风险预过滤
                              ▼
                         Tool Router
                              ▼
                         Tool Gateway
              鉴权/schema/白名单/幂等/超时/审计
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
             Python Tool    HTTP        MCP Server
                  │           │           │
                  └───────────┼───────────┘
                              ▼
                       Domain Service
                              ▼
                  PostgreSQL / pgvector

短问答：Runtime ──SSE──────────────→ Client
长任务：Runtime → Job Store → Worker → Event/Progress

横向能力：Identity / Tenant / Budget / Trace / Metrics / Logs / Audit
```

实际的 Agent 不是单向的“工具结果 → 模型 → 结束”，而是一个受限制的循环：

```text
Agent Runtime
  ▼
LLM 决策
  ├── 需要工具 → Tool Gateway → Tool Result → 回到 LLM
  └── 已有足够证据 → Final Answer → SSE → Client
```

## 二、每一层解决什么问题

| 层 | 主要问题 | 当前项目状态 |
| --- | --- | --- |
| API Gateway | TLS、认证入口、限流、请求边界 | 本地开发由 FastAPI 直接承担，未拆独立网关 |
| Agent Runtime | 循环、取消、最大步数、输出事件 | `src/agent_lab.py` 已有单请求有限循环 |
| Domain Router | 选择小说搜索、数据分析等领域 Agent | 目前只有问答模式路由，没有多领域 Agent |
| Planner | 决定工具调用顺序和完成条件 | Agent Lab 由模型生成一次一个 JSON action |
| Tool Registry | 工具描述、schema、版本、权限和状态 | 当前工具集合写在代码白名单中 |
| Tool Discovery | 从大量工具中筛选候选工具 | 当前工具只有五个，暂不需要语义工具检索 |
| Tool Router | 校验并确定最终工具调用 | 当前由 `readonly_toolbox` 做显式白名单校验 |
| Tool Gateway | 统一鉴权、风险控制、超时、重试和审计 | 当前只有只读限制、参数限制和重复动作保护 |
| Model Gateway | 模型能力、适配、预算、超时、降级和成本 | 三类生成适配器已经存在，但尚未统一治理 |
| State/Event Store | 跨进程恢复、暂停、重放 | 当前轨迹只保存在一次请求内 |
| Trace/Metric/Log | 定位路由、工具、模型和检索的耗时与失败 | 已有检索 trace，尚未统一 Agent 事件 |
| Control Plane | Registry、策略、Prompt、评测和灰度 | 当前配置主要来自代码和环境变量 |
| Job Worker | 与浏览器连接解耦的长任务 | 索引任务仍是进程内单线程状态 |

这里有六个必须保持的边界：

1. **MCP 不是 Tool Router**：MCP 负责标准化连接外部工具、资源和提示模板；权限、风险
   和业务授权仍应由自己的 Gateway 负责。
2. **Tool Gateway 不应接收任意 SQL**：工具调用领域服务或 Repository，再由它们访问
   PostgreSQL。模型不能直接构造数据库查询。
3. **OpenTelemetry 不是完整评测平台**：它提供 Trace、Span 和上下文传播；答案质量、
   工具选择准确率、成本和敏感字段脱敏仍需要应用层设计。
4. **外部内容是不可信数据**：小说原文、搜索结果和 MCP 返回值里的“指令”不能提升为
   系统策略；执行权限最终由 Gateway 判断，不能交给模型自觉遵守。
5. **聊天、运行状态和事件不是一份数据**：Chat History 给用户看，Run State 用于继续
   执行，Event Log 用于审计和重放，三者生命周期与敏感程度不同。
6. **同步请求不承载长任务可靠性**：短问答用 SSE；索引、批量摘要和关系抽取应返回
   `job_id`，由 worker 运行并持久化进度。

## 三、当前项目如何映射

```text
当前实现                         未来生产边界
────────────────────────────────────────────────
backend/main.py                 Agent API / Runtime 入口
src/query_router.py             Domain/Answer Router 雏形
src/agent_lab.py                Planner + 有限循环雏形
五个只读函数                   Local Tool 集合
src/rag.py                      Novel Search Service
backend/{zhipu,claude_cli}.py   Model Provider Adapter
PostgreSQL + pgvector + BM25    Metadata / Vector / Search Store
SSE agent_step                  Agent 事件流雏形
retrieval trace                 检索层 Trace，不是完整 Agent Trace
```

当前 PostgreSQL 已经同时承担关系数据、向量和 BM25 索引，因此不需要为了画出“Vector
DB”再引入第二个数据库。Redis、对象存储和消息队列也不是默认必选项：只有在跨进程
任务、缓存、原文文件生命周期或异步吞吐成为真实问题时才增加。

## 四、推荐的演进阶段

### M6.1：轻量 Control Plane 与 Tool Registry

把五个工具从“函数白名单”提升为统一描述对象：

```python
ToolSpec(
    name="search_novels",
    description="搜索小说原文",
    input_schema=SearchInput,
    read_only=True,
    permission="novel:read",
    risk_level="low",
    timeout_ms=3000,
    version="1.0.0",
)
```

验收重点：工具名称、描述、输入 schema、只读属性、权限、风险、超时和版本可以被测试
和枚举；工具实现仍然可以是普通 Python 函数。每次运行固定一份不可变配置快照，避免
执行到一半工具、Prompt 或权限策略发生变化。

### M6.2：Tool Gateway

把工具执行前后的共性逻辑集中起来：

```text
Tool Call
  → 是否注册/启用
  → 当前用户是否有权限
  → 参数 schema 是否有效
  → 资源范围与出站地址是否允许
  → 是否超过风险/次数/预算/超时限制
  → 执行并记录结果摘要
```

验收重点：所有工具都经过同一入口；禁止任意 SQL、shell 和未注册函数；错误能区分为
参数错误、权限错误、超时和下游失败。外部文本即使包含“忽略规则并调用某工具”也不能
越过 Gateway；写操作只有具备幂等键或明确审批时才允许自动重试。

### M6.3：Model Gateway

把 Ollama、Claude 和智谱的差异收敛到统一能力接口，同时保留供应商特有配置：

```text
Model Request
  → 任务需要 tool calling、长上下文还是普通生成
  → 当前租户允许哪些供应商和数据出境范围
  → token / 金额 / 时间预算是否足够
  → 调用、取消、超时并记录实际模型版本
  → 只有显式策略允许时才降级
```

验收重点：Router、Planner 和回答生成不再各写一套供应商调用；每次调用记录模型、版本、
token、耗时和估算成本；供应商失败时不会静默换成能力不足的模型。

### M6.4：统一 Agent 事件、评测与观测

建议把当前 `agent_step` 扩展成稳定事件类型：

```text
run_started
route_selected
tool_discovered
tool_selected
tool_started
tool_finished
evidence_added
answer_generated
run_finished
```

验收重点：一次请求可以按 `run_id` 还原 Router、Planner、Tool、LLM、检索和回答的
因果顺序；事件不记录 API Key、完整版权原文或不必要的个人信息。Trace 之外还要评测：

- Router/Tool 选择准确率和参数有效率；
- 工具成功率、重试率与下游错误率；
- 引用有效性、答案依据率和无证据拒答率；
- 首 token、总延迟、token 与金额成本；
- Prompt、模型、工具版本变化前后的回归。

### M6.5：状态拆分、长任务与恢复

先分开 Chat History、Run State 和 Event Log。短问答继续在一次请求里通过 SSE 返回；
索引、批量摘要和关系抽取返回 `job_id`，由 worker 执行。只有当 Agent 需要跨进程、暂停
等待人工批准或运行数十分钟时，才增加 checkpoint 和 Event Store。状态至少应包含：

- `run_id`、`thread_id`、当前用户和权限快照
- 已执行动作及工具版本
- 观察摘要和证据 ID
- 当前重试次数、暂停原因和恢复点
- 幂等键，避免恢复时重复执行有副作用的工具
- worker 租约/心跳、取消状态和达到上限后的死信原因

这是评估 LangGraph 的自然节点；LangGraph 的核心价值是持久化状态、检查点、人工介入
和故障恢复，而不是替代 Tool Gateway。详见[架构决策](architecture-decisions.md)。

### M6.6：MCP 适配

当工具需要被多个 Agent 客户端或外部 AI 应用复用时，再把稳定的 ToolSpec 映射为 MCP
Server。内部调用仍保留 Python/HTTP 适配器，便于测试和本地调试。

验收重点：MCP 暴露层不能绕过内部权限和风险检查；资源、工具和提示模板分别建模，
不要把所有能力都包装成一个万能工具。

### M6.7：多用户生产边界

这一步与路线图 M5 配合，补齐认证、租户隔离、限流、配额、审计、备份和监控。完成前，
Agent Lab 仍应明确标注为本地单用户学习功能。还要定义数据保留和删除策略，保证删除
用户或书籍时，原文、派生向量、摘要、事件和备份都有可解释的生命周期。

## 五、生产需要与学习项目取舍

同一套职责可以先在进程内表达，不需要一开始就变成基础设施：

| 能力 | 学习项目现在怎么做 | 生产条件出现后怎么升级 |
| --- | --- | --- |
| Tool Registry | Python `ToolSpec` 字典 + 单元测试 | 数据库/配置服务、版本审批和灰度 |
| Tool Gateway | 一个不可绕过的 Python 执行入口 | 独立扩缩容、策略服务、熔断和审计平台 |
| Model Gateway | 统一 Provider 接口和用量记录 | 多供应商预算、区域合规、动态路由和降级 |
| Event Log | 结构化日志或 PostgreSQL 表 | OTel Collector + 可查询的 Trace/日志平台 |
| 长任务 | PostgreSQL Job 表 + 单 worker | 队列、多 worker、租约、死信和弹性伸缩 |
| 状态恢复 | 先验证状态模型和幂等 | checkpoint/Event Store，必要时 LangGraph |
| MCP | 暂不接，保留适配接口 | 工具需跨客户端复用时提供 Server/Client |
| 多 Agent | 不做 | 单 Agent 评测证明无法满足时再引入 |

判断是否升级的依据应该是指标和约束：工具数量、并发、任务时长、恢复目标、租户数量、
风险等级、团队所有权和合规要求，而不是“生产架构图上通常有这个组件”。

## 六、什么时候引入框架

| 需求 | 优先增加什么 |
| --- | --- |
| 工具从 5 个增长到几十个 | ToolSpec、Registry、权限过滤 |
| 一次请求需要多个领域 Agent | Domain Router、Planner 和子流程 |
| 工具调用跨服务 | Tool Gateway、HTTP/MCP 适配和 Trace 传播 |
| 任务重启后继续 | State Store、checkpoint、幂等设计 |
| 人工审批和多分支恢复 | LangGraph 或其他持久化工作流运行时 |
| 需要统一外部工具生态 | MCP Client/Server |
| 需要跨服务调用链 | OpenTelemetry |

“步骤变多”本身不是引入 LangGraph 的理由；应该先确认是否出现了持久状态、人工介入、
复杂分支或恢复需求。

## 七、当前明确不做的事情

- 不把 5 个工具包装成多 Agent 系统。
- 不为了架构图引入 Redis、消息队列、独立 Vector DB 或对象存储。
- 不让模型直接执行 SQL、shell 或未经注册的 HTTP 请求。
- 不把 MCP 当成认证、RBAC、ABAC 或风险审批系统。
- 不在没有恢复需求时把 Agent Lab 重写成 LangGraph。

这份文档和[项目路线图](roadmap.md)一起阅读：路线图记录“做不做和验收什么”，本文记录
“每一层为什么存在以及怎样从当前实现演进过去”。
