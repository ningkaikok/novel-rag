# 架构决策：当前是否需要 LangGraph

> 状态：**暂不引入**
> 适用版本：当前本地单用户小说 RAG 学习项目
> 复审条件：Agent Lab 超出单请求、五步、只读工具的当前边界

## 先说结论

当前项目已经有一个小型 Agent Lab，但仍不需要 LangGraph，也不需要为了“技术栈更完整”
而加入它。

现在的一次问答是固定流水线：改写查询、检索、融合、重排、补邻居、生成、保存。
每一步的输入输出都可以从普通函数和 SSE 事件直接观察。对学习项目来说，这种显式
编排有一个重要优势：遇到错误时，可以明确判断是切分、召回、排序还是生成出了问题。

LangGraph 是面向**长时间运行、有状态、可能分支或暂停恢复的 Agent 工作流**的
低层编排框架。它擅长的持久化检查点、可恢复执行和人工介入都很有价值，但当前
项目还没有对应问题。当前五个只读工具、最多五步的状态可以由一个普通 `for` 循环完整
表达；此时引入只会增加状态定义、节点/边、检查点和调试抽象。

## 先分清两个容易混淆的名字

| 名称 | 解决的问题 | 本项目状态 |
| --- | --- | --- |
| **GraphRAG** | 把人物和关系做成知识图，回答全书范围的聚合问题 | 有实验原型，因关系误报较多而默认关闭 |
| **LangGraph** | 把模型、工具和业务步骤编排成可持久恢复的有状态工作流 | 当前未使用，也不是 GraphRAG/Agent 的前置条件 |

`src/graph.py` 实现的是第一种。文件名里有 `graph`，不代表项目已经使用或必须使用
LangGraph。

## 当前调用链为什么普通 Python 足够

```text
POST /api/ask
  → 模式覆盖或保守规则选择回答路径
  ├─ 自由问答 → 跳过书架索引
  └─ 原文问答
       → 可选：根据历史补全追问
       → 检索范围判断
       → 可选层级摘要导航 + 向量 / BM25 / 结构性召回
       → RRF 融合
       → 可选：交叉编码器重排
       → 邻居扩展
  → 按模型类型生成
  → SSE 推送并保存对话
```

标准 RAG 这条链路有少量条件分支，但没有 Agent 式的不确定控制流：

- 不会让模型自主决定下一步调用哪个工具；
- 没有“调用工具 → 观察结果 → 再规划”的循环；
- 一次 HTTP 请求内完成，不需要隔天或跨进程恢复；
- 用户中断只需停止上游生成器，不需要保存整条工作流的检查点；
- 会话历史只是查询改写的输入，PostgreSQL 已能满足，不等于工作流状态持久化。

新增的问答路由仍是一次确定性分类，没有模型工具循环；规则、原因和模式覆盖详见
[问答模式与自动路由](answer-routing.md)。它让普通函数编排的边界更清晰，并没有改变
“暂不需要 LangGraph”的结论。

M2 的后台增量索引同样没有改变这个结论。它是固定的“扫描 → 切分 → Embedding →
BM25 → 单书事务”循环：`index_manifest` 已经是最小检查点，后端重启后重新扫描即可
跳过完成的书。这里用一个线程和 PostgreSQL 事务比引入通用工作流运行时更透明；只有
未来需要在单本书内部跨进程恢复昂贵节点、人工审批或多 worker 调度时才复审。

因此当前的 `retrieve_hybrid_stream()` 本身就是最小、透明的编排器。把它拆成图节点
不会自动提高召回率，也不会修复 GraphRAG 的关系抽取质量。

## Agent Lab 为什么普通循环仍然足够

`POST /api/agent/ask` 确实包含不确定控制流：模型在五个工具里选择一个，工具结果写回
观察，再规划下一步。但它的边界非常窄：

```text
for step in range(max_steps):                 # max_steps 只能是 3～5
    action = planner(question, observations)  # 一次只返回一个 JSON action
    result = readonly_toolbox.execute(action) # 显式白名单
    observations.append(result)
    if action == answer_with_citations: break
```

- 状态只是问题、观察列表和本次请求内的证据表；
- 所有工具同步、只读，没有人工审批和副作用补偿；
- 一次 HTTP 请求内结束，不承诺刷新或重启后的恢复；
- 重复动作、无证据回答和步数上限都能用几行普通 Python 明确测试。

所以它适合用来先学会 Agent 的基本构造。若现在换成框架，初学者反而要先理解图状态、
节点消息和运行时，才能看见本来只有一个循环的行为。完整边界见 [Agent Lab](agent-lab.md)。

## 何时应该引入

出现以下任意一种真实需求时，再做一个小型验证分支：

1. **复杂工具图**：工具数量和分支显著增加，需要并行子任务、子图或多 Agent 协作，
   普通循环已很难读懂。
2. **可恢复长任务**：索引、摘要或 Agent 执行要运行数十分钟，进程重启后必须从上一个
   成功节点继续，而不是整批重跑。
3. **人工介入**：关系抽取写库、删除索引或发布答案前，要暂停并等待用户审核，然后
   从原状态继续。
4. **复杂路由和重试**：不同问题类型进入不同子流程，节点有明确的重试、回退和循环，
   普通 `if/try` 已经难以理解和测试。
5. **需要工作流级观测**：要查看每个节点的持久状态、回放某次执行，或从中间状态
   派生另一条执行路径。

如果只是增加一种召回方式，优先在现有多路召回中加一个函数并纳入评测；“步骤变多”
本身不是引入框架的理由。

## 生产化时不要把框架当成全部架构

如果项目从单用户学习 demo 走向生产，新增的重点不应是把所有代码改写成图，而是先
明确以下边界：

```text
Domain Router → Planner → Tool Discovery → Tool Router → Tool Gateway → Tool
```

- Router 决定交给哪个领域流程；Planner 决定执行计划。
- Tool Discovery 只负责从注册表筛选候选工具，并先做权限和状态过滤。
- Tool Router 负责结构化选择，Tool Gateway 负责最终鉴权、schema、风险、超时、重试和审计。
- MCP 是外部工具连接协议，不替代 Gateway 的权限边界。
- OpenTelemetry 能串起 Trace/Span，但不会自动提供答案评测、成本控制或敏感信息脱敏。

当前项目的五个工具太少，暂时用代码白名单表达这些概念更适合学习。真正开始 M6 时，
先实现 `ToolSpec`、Registry、Gateway 和稳定事件，再决定是否增加 MCP 或 LangGraph。
目标架构、迁移阶段和验收标准见 [Agent 平台化架构](agent-platform-architecture.md)。

## 将来迁移时怎样控制风险

不要一次把整个后端重写成图。先保留现有纯函数，只把编排层替换掉：

| 图节点候选 | 继续复用的现有能力 |
| --- | --- |
| `rewrite_query` | `src/query_rewriter.py` |
| `retrieve` | `NovelRAG.retrieve_hybrid_stream()` 内的召回函数 |
| `rerank` | `src/reranker.py` |
| `build_context` | `expand_neighbors()` / `build_prompt()` |
| `generate` | Ollama、Claude CLI、智谱三个生成适配器 |

先用相同评测集比较“迁移前后检索结果是否一致”，再考虑检查点和人工审批。这样学习到
的是编排框架解决了什么问题，而不是得到一份行为已经改变、却不知道为什么的重写。

## 注释和文档约定

这个项目的注释重点解释**为什么这样设计、失败时怎样降级、有哪些代价**，不逐行翻译
代码。功能变化后至少同步检查三处：

1. 源码模块 docstring 和关键分支注释；
2. `docs/code-walkthrough.md` 的调用链与完成状态；
3. `README.md` 的启动方式、配置项和学习入口。

文档中的实验结论必须能指向评测脚本或历史基线；不能核验的行业数字不作为项目结论。

## 延伸阅读

- [LangGraph 官方概览](https://docs.langchain.com/oss/python/langgraph/overview)：框架定位、
  durable execution、streaming、human-in-the-loop 和 persistence。
- [LangGraph 官方持久化说明](https://docs.langchain.com/oss/python/langgraph/persistence)：
  检查点、线程、恢复执行和 time travel 的使用场景。
