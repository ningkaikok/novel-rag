# 系统技术架构（2026-08 现状）

> 本文是当前代码结构的"一张图"版本：整体分层、一次问答的完整链路、索引流水线、
> 数据模型和工程化设施。各部分的深入讲解见对应专题文档（每节末尾有链接）。
> 图使用 mermaid，GitHub 上可直接渲染。

## 一、整体架构

```mermaid
flowchart TB
    subgraph client["浏览器（单端口 8000，生产由 FastAPI 托管 dist）"]
        UI["React 18 + Vite + TS + antd<br/>App.tsx（组件树）<br/>useChatStream / useBookshelf hooks<br/>lib/streaming.ts 打字机纯函数"]
    end

    subgraph api["FastAPI（backend/）"]
        MW["RequestIDMiddleware<br/>request_id 注入日志 + 访问日志"]
        R1["POST /api/ask · /api/agent/ask（SSE）"]
        R2["书架 CRUD + 索引任务<br/>search / sessions / model / health"]
    end

    subgraph core["核心业务（src/，不依赖 Web 框架）"]
        ROUTER["query_router 三模式"]
        RAG["rag.py 编排层<br/>NovelRAG = RetrievalMixin + GenerationMixin"]
        MIX["retrieval_mixins 向量/BM25/结构性<br/>hierarchy 层级导航<br/>reranker 交叉编码器<br/>confidence 低置信度信号<br/>query_expander 自适应扩展(默认关)"]
        GEN["generation_mixin Prompt+流式生成"]
        AGENT["agent_lab 五个只读工具循环"]
    end

    subgraph gen["生成模型（可切换）"]
        OLLAMA["Ollama 本地 qwen2.5"]
        CLAUDE["Claude CLI（复用订阅）"]
        GLM["智谱 GLM"]
    end

    subgraph data["PostgreSQL + pgvector"]
        CH["novel_chunks + HNSW"]
        TERMS["chunk_terms（BM25）"]
        HIER["hierarchy_summaries/_manifest"]
        MANI["index_manifest(+quality_report)"]
        CHAT["chat_turns(+run_config)"]
        TASKS["index_task_runs"]
    end

    UI -->|"api.ts：REST=OpenAPI 生成类型<br/>SSE 事件手写协议"| MW
    MW --> R1 --> ROUTER --> RAG
    R2 --> TASKS
    RAG --> MIX --> CH & TERMS & HIER & MANI
    RAG --> GEN --> OLLAMA & CLAUDE & GLM
    RAG -->|"检索评测 trace 逐层落库"| CHAT
    R1 --> AGENT --> MIX
```

要点：
- **src/ 不 import FastAPI**——评测脚本直接调 `NovelRAG`，Web 只是薄封装；
- **前后端类型契约**：`schemas.py` → `openapi.json` → `api-generated.ts`，
  CI drift 检查保证不漂移；
- **生成模型三通道**在 Web 层按前缀路由（`glm:` / `claude:` / 其余走 Ollama），
  src 层只认注入的生成函数。

## 二、一次原文问答的链路（标准 RAG 模式）

```mermaid
sequenceDiagram
    participant U as 用户
    participant F as FastAPI
    participant Q as query_router
    participant W as query_rewriter
    participant N as NovelRAG 编排
    participant V as 向量召回(pgvector)
    participant B as BM25(chunk_terms)
    participant H as hierarchy 层级
    participant R as bge-reranker
    participant C as confidence
    participant L as LLM

    U->>F: 问题（SSE 流式）
    F->>Q: auto / grounded / free？
    Q-->>N: 小说问题→走检索；开放问题→直接问模型；拿不准→保守检索
    N->>W: 多轮？最近 4 轮×200 字补全指代
    par 多路并行召回
        N->>V: embedding(bge-small-zh) HNSW Top-20
        N->>B: jieba 分词 BM25 Top-20
        N->>H: 全局问题先章节/全书摘要定位再回原文
    end
    Note over N: RRF 融合去重 → 候选池
    N->>R: 交叉编码器精排（用 indexed_text 含上下文增强）
    N->>C: 归一化分差/词覆盖/跨书分散度
    alt 低置信且 QUERY_EXPAND_ENABLED
        C-->>N: 生成 ≤3 个改写 → 补救检索一次 → 重排（结构上只此一次）
    end
    N->>N: expand_neighbors/chapter 组装证据（[n] 编号）
    N->>L: PROMPT_TEMPLATE(v1) 流式生成
    L-->>U: token 逐字 + sources + trace（四层排名变化）
    Note over F: 全程落库 chat_turns（含 run_config 快照）<br/>停止=后端真停；刷新恢复历史
```

失败边界：任何一步无证据 → 明确拒答；补救重排失败 → 退回原结果；
trace 里能回答「正确片段在哪一步丢的」（召回/融合/精排/生成）。

## 三、索引流水线（增量 + 质量门禁）

```mermaid
flowchart LR
    A["data/novels/*.txt"] --> P{"plan_index<br/>SHA-256 + 配置指纹"}
    P -->|新增/修改| L["loader UTF-8→GB18030"]
    P -->|未变化| SKIP["跳过（秒级扫描结束）"]
    L --> C["段落聚合切分 500/80"]
    C --> G{"index_quality 门禁<br/>真实 tokenizer"}
    G -->|"空输入/截断/坏向量"| X["硬错误：<br/>阻止本书替换，旧索引保持可用"]
    G -->|通过| E["bge-small-zh embedding"]
    G -->|通过| T["jieba BM25 词频"]
    E & T --> TXN["单书事务原子替换<br/>向量+BM25+manifest 同事务"]
    C -.->|可选 CONTEXTUAL_MODE=auto| CTX["LLM 补上下文说明<br/>内容哈希缓存复用"]
    C -.->|层级指纹独立| HIER["章节/全书摘要"]
    C -.->|GRAPH_ENABLED| GRAPH["人物关系边"]
    TXN --> DONE["index_task_runs 落库<br/>任务卡片可跨重启恢复"]
```

一致性保证：取消/异常时**当前书**事务回滚，其他书不受影响；层级摘要算法变化只补
摘要不重算基础向量。质量门禁实测拦下过 chunk800 这类会被静默截断的配置。

## 四、数据模型

| 表 | 角色 | 关键点 |
|---|---|---|
| `novel_chunks` | 片段 + 向量 + 章节元数据 | 回答引用的事实来源 |
| `chunk_terms` | BM25 词频倒排 | 与向量同事务切换 |
| `index_manifest` | 文件哈希 + 流水线指纹 + quality_report | 增量同步的检查点 |
| `hierarchy_summaries/_manifest` | 章节/全书导航摘要 | 只做导航，证据必须回原文 |
| `chunk_contexts` / `graph_characters` | Contextual/关系图缓存 | 内容哈希主键，增量复用 |
| `chat_turns` | 会话历史 + trace + agent_steps + run_config | 刷新恢复、在线配置快照 |
| `index_task_runs` | 索引任务快照 | 重启后侧栏恢复卡片 |

## 五、工程化设施

```mermaid
flowchart LR
    subgraph local["本地开发"]
        UV["uv（uv.lock 锁定）"]
        PC["pre-commit：ruff · pyright 增量门禁 · eslint · prettier"]
    end
    subgraph ci["CI（每次 push/PR）"]
        FE["前端 job：eslint → prettier → npm audit → vitest+cov → tsc → 类型契约 drift"]
        BE["后端 job：pip-audit → ruff → pyright 基线门禁 → pytest-cov(75%)"]
        E2E["e2e job：playwright ×18（mock API，无需真实依赖）"]
        NOTIFY["飞书通知"]
    end
    subgraph nightly["夜间/手动"]
        EVAL["检索评测门禁：原创小语料临时库重建 → strict 对照基线，回退即标红"]
    end
    subgraph deploy["部署"]
        DOCKER["多阶段 Dockerfile → compose（pgvector + 缓存卷）"]
    end
    local --> ci --> deploy
    nightly -.->|指标守门| ci
    DEPEN["Dependabot 周更<br/>pip/npm/actions"] -.-> ci
```

质量数字（2026-08-22）：pytest 219 · vitest 19 · playwright 18 · pip-audit/npm audit 清零 ·
pyright 存量基线 35 条只拦增量 · 后端覆盖率 75%（观测基线）。

## 六、设计决策速查

| 决策 | 结论 | 出处 |
|---|---|---|
| 为什么不用 LangChain/LangGraph | 先看清数据流；触发条件到了再引入（M6.5 前） | [architecture-decisions](architecture-decisions.md) |
| 为什么留在 PostgreSQL | 中小规模 pgvector 即业界默认；混合检索同库同事务 | [rag-techniques](rag-techniques.md) |
| 为什么 monorepo 目录不拆 apps/ | 单产品拆目录是化妆性改动；类型契约用 OpenAPI codegen 解决 | 工程化阶段讨论记录 |
| Contextual Retrieval 为何默认 auto | 实测 4.4s/条；小书自动、大部头闸门拦截、后端可用性预检 | roadmap M3.4 |
| Judge 忠实度为何不上线 | 53 条标注集一致率最高 67.9% < 80% 门槛，影子调用 | [校准报告](experiments/m35-faithfulness-calibration.md) |
