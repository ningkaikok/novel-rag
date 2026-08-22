# 小说 RAG 查询系统

[![CI](https://img.shields.io/github/actions/workflow/status/ningkaikok/novel-rag/ci.yml?branch=main&label=CI&logo=github)](https://github.com/ningkaikok/novel-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ningkaikok/novel-rag?label=release&color=orange)](https://github.com/ningkaikok/novel-rag/releases/latest)
[![License](https://img.shields.io/github/license/ningkaikok/novel-rag?label=License&color=blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](pyproject.toml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
[![React](https://img.shields.io/badge/React-18-20232a?logo=react&logoColor=61DAFB)](frontend/package.json)

> 一个把 RAG 做「透明」的中文小说问答系统：每次回答都能逐层展开向量、BM25、RRF 融合与重排的候选排名、分数变化和耗时，看清答案是怎么被找出来的 —— 以及在哪一步被弄丢的。

基于本地向量检索的小说问答 demo。检索和 Embedding 全部在本机运行；生成模型默认用本地 Ollama（不需要任何外部 API Key），也可以按需切换到你自己的 Claude 订阅或智谱 GLM（见下文“切换生成模型”）。

## 四个和一般 RAG demo 不太一样的地方

1. **混合检索是完整的一条链，不是只算一次向量相似度。**
   段落聚合切分（约 500 字、片段间重叠）→ 向量召回（`bge-small-zh-v1.5`）+ BM25 → RRF 融合 → 重排；另建章节与全书摘要做层级检索，主题类和跨书比较类问题先定位书与章节，再回到原文参与融合。

2. **检索过程可观测，也可评测。**
   界面里能展开每一层的候选排名与分数变化，用来定位「答案不对，是哪一步把证据丢了」；`tests/` 维护问答评测集与历史基线，改检索策略前后能直接对比回归，而不是凭感觉调参。

3. **索引是增量的，而且保证一致。**
   按小说文件 SHA-256 与索引配置指纹识别变化，只重建新增、修改或删除的书；每本书的向量与 BM25 索引在同一事务内原子切换，不会出现「向量更新了、BM25 还是旧的」这种中间态。

4. **默认全本地零成本，也能随时换云端模型。**
   默认 Ollama（`qwen2.5:7b`）+ 本地 Embedding，不需要任何 API Key；需要更强生成质量时可在界面里切到 Claude 或智谱 GLM。另有独立的 **Agent Lab**：不依赖任何 Agent 框架手写的有限步只读工具循环，通过 SSE 逐步展示工具选择、结果与证据。

技术栈：
- **文本切分**：按段落聚合成约 500 字的片段，片段间有重叠，避免切断语义。
- **Embedding**：`sentence-transformers`，本地模型 `BAAI/bge-small-zh-v1.5`（中文效果较好，体积约 95MB）。
- **数据库**：PostgreSQL + pgvector，片段、原文和 embedding 存储在 `novel_rag` 数据库的 `novel_chunks` 表中。
- **增量索引**：按小说文件 SHA-256 和索引配置指纹识别变化，只更新新增、修改或删除的书；每本书的向量与 BM25 在同一事务中原子切换。
- **层级检索**：为片段建立章节摘要和全书摘要；主题、成长、跨书比较先定位书与章节，再回到原文参与融合和重排。
- **检索可观测性**：每次问答可展开查看向量、BM25、RRF、重排的候选排名、分数变化与耗时。
- **Agent Lab**：独立的 3～5 步只读工具循环，逐步展示选择、工具结果与证据，不依赖 Agent 框架。
- **生成模型**：默认本地 [Ollama](https://ollama.com)（`qwen2.5:7b`），界面里可随时切换到其他本地模型、你自己的 Claude 订阅或智谱 GLM（云端模型会发送问题和召回片段，见下文）。
- **后端**：FastAPI（`backend/main.py`），把检索/生成逻辑包成 HTTP 接口，回答用 SSE 逐字流式返回。
- **前端**：React + Vite + TypeScript + Ant Design（`frontend/`），书卷气界面「书虫」。用 antd 组件 + `ConfigProvider` 主题令牌保留藏青主色与暖底书卷气，支持浅色/深色主题、上传/删除书籍、示例问题、流式回答，以及可点击的原文出处引用。前后端类型契约由 OpenAPI 生成（`schemas.py` → `openapi.json` → `api-generated.ts`，CI 有 drift 检查），改 Pydantic 模型后跑 `uv run python scripts/export_openapi.py && cd frontend && npm run gen:api`。
- **对话体验**：生成中可以点「停止」——不只是前端不再显示新字，后端会真的停止向模型索取内容（用云端模型时不多花钱）；刷新页面或重开浏览器后，之前的问答、原文出处、思考过程会自动恢复，中途被停止的那轮也会如实标出来；往上翻看历史时，下面来了新回答会提示「有新回复」，不会悄悄错过。

## 目录结构

```
novel-rag/
├── src/           # 核心业务逻辑（切分/入库/检索/重排），不依赖 Web 框架
├── backend/       # FastAPI 后端（只做"把 src 包成 HTTP"这一件事）
├── frontend/      # React + Vite 前端（类型由 OpenAPI 生成，见 api-generated.ts）
├── scripts/       # 独立工具（检索评测、质量门禁、OpenAPI 导出等）
├── docs/          # 学习文档与实验报告，见下
├── tests/         # pytest + 问答评测集与历史基线
├── data/novels/   # 放小说 .txt 文件
├── Dockerfile     # 多阶段构建：前端 dist + Python 运行时，单镜像部署
└── docker-compose.yml  # 应用 + pgvector 一键起（模型缓存/文本挂载持久化）
```

## 📖 学习路线

这个项目也是一份 RAG 学习材料，每个技术点都配了**真实的失败案例和实测数据**，
不是纸上谈兵：

| 文档 | 讲什么 | 什么时候看 |
| --- | --- | --- |
| [**系统技术架构**](docs/system-architecture.md) | 当前全栈架构图：分层、问答链路、索引流水线、数据模型、工程化设施 | ⭐ 想快速建立全局认识 |
| [**RAG 学习总览**](docs/rag-overview.md) | 四个杠杆是什么、完整链路、**所有实测数据汇总（含负面结果）**、方法论教训 | 从这里深入 RAG |
| [**代码导读**](docs/code-walkthrough.md) | 这份代码怎么读、建议的阅读顺序、可直接上手跑的实验 | 第一次接触这个项目 |
| [**RAG 核心技术**](docs/rag-techniques.md) | 检索评测、BM25、重排、Contextual Retrieval、多轮改写、GraphRAG 的原理与实测 | 想深入某个具体技术 |
| [**问答模式与自动路由**](docs/answer-routing.md) | 一个输入框如何区分开放问题与小说问题，以及怎样离线评测 | 想理解新增的路由层 |
| [**章节元数据与可核验引用**](docs/citations-and-chapters.md) | 章节识别、`[n]` 引用定位、兼容迁移和引用评测 | 想理解回答如何回到原文证据 |
| [**增量索引与后台任务**](docs/incremental-indexing.md) | 文件哈希、单书事务、进度、取消和失败重试 | 想理解长任务如何保持数据一致性 |
| [**M3 层级检索**](docs/hierarchical-retrieval.md) | 片段→章节→全书、摘要导航、原文回溯和跨书公平召回 | 想处理主题与人物成长问题 |
| [**检索可视化评测**](docs/retrieval-observability.md) | 向量/BM25/RRF/重排排名怎样逐层变化，如何定位丢失点 | 想调试检索质量 |
| [**检索是否真的在起作用**](docs/grounding-verification.md) | 用对照实验区分「模型凭记忆答对」和「真的用了检索片段」 | 想验证检索有效性 |
| [**Agent Lab**](docs/agent-lab.md) | 五个只读工具、有限步循环、安全边界和 SSE 轨迹 | 想从普通 Python 学 Agent |
| [**架构决策：是否需要 LangGraph**](docs/architecture-decisions.md) | 当前请求链路为什么保持显式编排、GraphRAG 和 LangGraph 的区别、什么情况下再引入 | 想学习技术选型 |
| [**Agent 平台化架构**](docs/agent-platform-architecture.md) | Control/Data Plane、Tool/Model Gateway、安全、状态、Trace、MCP 与生产演进 | 想从 Agent Lab 走向生产设计 |
| [**项目路线图**](docs/roadmap.md) | 已完成能力、后续里程碑和每阶段验收标准 | 想继续完善项目 |
| [**检索实验报告**](docs/experiments/m34-retrieval-matrix.md) | chunk 粒度与 embedding 模型对照的实测数据（含大部头验证） | 改检索参数前先看结论 |

另有两份面试向的整理：[流式中断与 UI 性能](docs/streaming-interview-notes.md)、
[从项目里提炼的 28 道面试题](docs/interview-questions.md)。

> 当前项目**不依赖 LangChain/LangGraph**。这是有意的架构选择，不是遗漏：先用普通
> Python 函数看清切分、召回、融合、重排和生成的数据流。Agent Lab 也故意先用最多
> 五步的普通循环展示基本原理；当它需要跨进程恢复、人工审批或复杂并行分支时，再引入
> LangGraph。Tool Registry、Tool Gateway、MCP 和生产级 Trace 的演进边界见
> [Agent 平台化架构](docs/agent-platform-architecture.md)，判断依据见上面的架构决策文档。

## 环境准备

1. 安装 [Ollama](https://ollama.com) 并拉取一个模型（推荐 7b，效果更好；机器配置一般可以换成 `qwen2.5:3b` 甚至 `1.5b`）：

   ```bash
   brew install ollama
   brew services start ollama
   ollama pull qwen2.5:7b
   ```

2. 安装 PostgreSQL 和 pgvector，并创建数据库（如果使用本机 Homebrew 安装）：

   ```bash
   brew install postgresql@18 pgvector
   brew services start postgresql@18
   createdb novel_rag
   psql -d novel_rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'
   ```

   如果数据库地址、用户名或密码不同，设置 `DATABASE_URL`，例如：

   ```bash
   export DATABASE_URL=postgresql://user:password@127.0.0.1:5432/novel_rag
   ```

3. 安装 [uv](https://docs.astral.sh/uv/) 并同步 Python 依赖（**需要 Python 3.11/3.12/3.13**，3.14 太新，`tokenizers` 等库还没有对应的预编译包）。`uv sync` 会按 `uv.lock` 锁定版本创建 `.venv`，保证和 CI 完全一致：

   ```bash
   # macOS: brew install uv；或 curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   source .venv/bin/activate
   ```

   开发工具（ruff / pyright / pre-commit）在 dev 依赖组里，`uv sync` 一并装好。提交前建议启用 Git 钩子：`uv run pre-commit install`。

4. 安装前端依赖（需要 Node.js 20.19+，推荐 22——vite 8 与 playwright 1.62 的硬要求；`nvm use 22`）：

   ```bash
   cd frontend
   npm install
   cd ..
   ```

## 添加你自己的小说

有两种方式，任选其一：

- **网页上传（推荐）**：启动网页后选择 `.txt` 文件。文件保存后会自动在后台建立增量索引，侧栏显示切分、Embedding、BM25 和入库进度；无需再手动全库重建。
- **手动放置**：把 `.txt` 文件放进 `data/novels/` 目录，再运行 `python src/ingest.py`。

每个文件当作一部小说，文件名会作为来源标注。目录里已经有一篇原创的示例短篇小说 `雾隐山庄.txt`，用于快速验证整个流程，可以直接删掉换成你自己的小说文本。

## 启动（FastAPI + React，推荐）

需要开两个终端：

**终端 1 — 后端 API（端口 8000）：**

```bash
source .venv/bin/activate
uvicorn backend.main:app --port 8000
```

**终端 2 — 前端 dev server（端口 5173）：**

```bash
cd frontend
npm run dev
```

浏览器打开 `http://localhost:5173`。左侧「我的书架」可上传/删除小说、查看后台索引进度、安全停止或重试任务，并在「更多设置」里检查文件变化和调整参考原文数量。底部保持一个输入框；工作模式可选择稳定的“标准 RAG”或教学用“Agent Lab”。标准 RAG 提供三种回答模式：

- **自动判断**：开放问题直接问模型，小说问题先检索原文；拿不准时保守地检索。
- **仅依据原文**：强制执行混合检索、重排和引用，资料不足就明确拒答。
- **自由问答**：不搜索小说书架，适合技术概念、写作、翻译和闲聊；没有索引也能用。

实际采用的路径和原因会显示在「思考过程」中。原文问答命中片段后会自动带入同一本书前后相邻片段，回答逐字流式显示；正文中的 `[1]` 可以点击并定位到对应的书名、章节名和片段编号，方便核实依据。前端 `/api` 请求由 Vite 代理到后端 `127.0.0.1:8000`。

对主题、人物成长和跨书比较问题，标准 RAG 会自动增加层级摘要导航；回答下方的
“检索评测”可逐层查看排名。切到 Agent Lab 后，界面会改为展示最多五次
“选择工具 → 获得观察 → 继续判断”，最终仍只能依据可点击的原文出处回答。

> 生产部署（Docker，推荐）：
>
> ```bash
> docker compose up --build -d
> # 把 .txt 小说放进 data/novels/ 后建索引：
> docker compose exec api uv run python src/ingest.py
> ```
>
> 打开 `http://localhost:8000`——单端口同时服务 API 与前端页面。PostgreSQL(pgvector)、
> 模型缓存、小说文本分别持久化在独立卷/挂载里；容器内访问宿主机 Ollama 走
> `host.docker.internal`（Linux 由 compose 的 host-gateway 提供）。
>
> 不用 Docker 的手动路径：`cd frontend && npm run build` 生成静态文件后，
> FastAPI 会自动检测 `frontend/dist` 并托管（存在即挂载，开发模式不受影响）。

## 切换生成模型（本地 Ollama / Claude 订阅 / 智谱 GLM）

输入框上方的模型下拉框支持三类来源，选完立即生效，不需要重启服务：

- **💻 本地（Ollama，完全离线）**：自动列出 `ollama list` 里已安装的模型（如 `qwen2.5:3b`、`qwen2.5:7b`）。
- **☁️ 我的 Claude 订阅（云端）**：如果本机装了 [Claude Code CLI](https://claude.com/claude-code) 并已登录，会额外出现 `haiku`/`sonnet`/`opus` 三档，**不需要单独配置 `ANTHROPIC_API_KEY`**——直接复用你本地已登录的 Claude 订阅（后端通过 `claude --print` 非交互调用）。
- **☁️ 智谱 GLM（云端）**：设置了环境变量 `ZHIPU_API_KEY` 时出现 `glm-4-flash`/`glm-4.5-air`/`glm-4.5`/`glm-4.6` 四档。

选择任何云端模型时请注意（界面上的胶囊标签和 Tooltip 也会显示同样的提示）：

- **不再是"完全本地"**：检索到的原文片段和你的问题会发送到对应厂商（Anthropic / 智谱）的服务器。
- **计入你自己的账号用量**，不是免费的（`glm-4-flash` 除外）。
- Claude 这条路径由于 CLI 没有"跳过 CLAUDE.md/记忆加载但仍用 OAuth 登录"的组合选项，回答风格可能会受你本机全局 `~/.claude/CLAUDE.md` 配置的轻微影响（例如强制用中文回复）。

### 配置智谱 GLM 的 API Key

Key **存在项目根目录的 `.env` 里，后端启动时自动加载**（`backend/dotenv_lite.py`，零依赖），不需要每次手动 export，也不会进版本库（`.env` 已被 `.gitignore` 忽略）：

```bash
cp .env.example .env   # 然后编辑 .env 填入你的 key
uvicorn backend.main:app --port 8000
```

启动日志会打印一行 `[env] 已从 .env 加载：ZHIPU_API_KEY`（只打变量名、不打值）确认生效。

优先级：**真实环境变量 > `.env`**，所以想临时换个 key 或模型，启动前 export 同名变量即可覆盖：

```bash
ZHIPU_API_KEY=另一个key uvicorn backend.main:app --port 8000
```

没配置这个 key 时，下拉框里不会出现 GLM 分组，其他功能不受影响。Key 在[智谱开放平台](https://open.bigmodel.cn/usercenter/apikeys)申请和吊销——**不要把 key 贴到聊天、issue 或截图里**，一旦泄露立即到该页面吊销重建。

实现细节见 [backend/claude_cli.py](backend/claude_cli.py) 和 [backend/zhipu.py](backend/zhipu.py)。

## 单独同步索引（命令行）

如果你是手动往 `data/novels/` 放的文件，或想在命令行同步：

```bash
source .venv/bin/activate
python src/ingest.py
```

正式写入前也可以先做一次不改数据库的质量预检：

```bash
python scripts/check_index_quality.py --novel data/novels/雾隐山庄.txt
```

预检会使用当前 Embedding 模型的真实 tokenizer 检查输入 token 长度，并报告编码方式、
空/重复片段、章节覆盖和乱码提示。索引同步时会再次执行同一门禁；超长输入、空内容、
异常向量或维度不一致会阻止当前书替换旧索引。质量摘要会随 `index_manifest` 保存，
不包含小说原文。

这一步会比较文件哈希和 `index_manifest` 清单，只处理发生变化的书。准备新数据时旧索引
仍可查询；最后在一个 PostgreSQL 事务里同时替换该书的向量、BM25 和清单记录。
任务失败或被取消时当前书自动回滚，其他书不受影响。

从 M2 之前的版本升级时，第一次运行会把现有书逐本迁移并建立哈希清单；以后没有
变化时只扫描文件就结束。网页的 `POST /api/reindex?force=true` 可用于明确要求全部重做，
平时不要使用强制模式。

从 M3 之前的版本升级时，文件未变化的书只会补建章节/全书摘要，不会重算基础片段
向量和 BM25。层级摘要使用独立清单，后续只改摘要算法也能单独迁移。

## 可调参数（环境变量）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地 embedding 模型（双编码器，负责快速粗筛） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | 本地重排模型（交叉编码器，约 1.1GB，首次使用自动下载） |
| `RERANK_ENABLED` | `1` | 设成 `0` 关闭重排（模型下载不了、或想对比重排前后效果时用） |
| `CHUNK_SIZE` | `500` | 每个片段的字符数上限 |
| `CHUNK_OVERLAP` | `80` | 相邻片段的重叠字符数 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 生成用的模型名 |
| `DATABASE_URL` | 当前用户本机 `novel_rag` | PostgreSQL 连接字符串 |
| `TOP_K` | `3` | 送进 prompt 的片段数（默认值是实测出来的，见 docs/rag-techniques.md 第 5 节） |
| `FULL_TEXT_MAX_CHARS` | `8000` | 点名的书全文小于这个字数时跳过检索、直接给模型全文 |
| `RECALL_K` | `20` | 关键词和向量检索各自召回的候选片段数量 |
| `CONTEXT_NEIGHBORS` | `1` | 问答时每个命中片段前后额外带入的相邻片段数 |
| `HIERARCHY_ENABLED` | `1` | 对全局问题开启章节/全书摘要导航；最终证据仍是原文 |
| `HIERARCHY_SUMMARY_MAX_CHARS` | `800` | 单个抽取式摘要的最大字符数 |
| `HIERARCHY_UNTITLED_CHUNKS` | `12` | 无章节标题小说的虚拟章节窗口大小 |
| `HIERARCHY_TOP_K` | `6` | 全局问题最多选择的章节摘要节点数 |
| `BM25_K1` | `1.2` | BM25 词频饱和速度，越小饱和越快（一般不用动） |
| `BM25_B` | `0.75` | BM25 文档长度归一化强度，0=不归一化、1=完全按长度惩罚（一般不用动） |
| `CONTEXTUAL_MODE` | `auto` | 上下文增强三档：`off` 关闭 / `auto` 小体量书默认构建、大部头跳过且要求生成后端可用 / `on` 强制（仍受片段数上限拦截） |
| `CONTEXTUAL_ENABLED` | 空 | 旧开关，兼容保留：设 `1` 等价 `CONTEXTUAL_MODE=on`；设 `0` 且未另设 MODE 时等价 `off` |
| `CONTEXTUAL_MAX_CHUNKS_PER_BOOK` | `2000` | 成本闸门：超过这个片段数的书直接跳过，不做上下文增强 |
| `CONTEXTUAL_MODEL` | `glm:glm-4-flash` | 生成上下文说明用的模型（用便宜的小模型就够） |
| `CONTEXTUAL_WORKERS` | `8` | 生成上下文的并发数 |
| `QUERY_REWRITE_ENABLED` | `1` | 多轮追问时先补全指代再检索；设成 `0` 关闭 |
| `QUERY_REWRITE_MODEL` | `glm:glm-4-flash` | 改写用的模型（用便宜快速的小模型，别用推理型大模型） |
| `QUERY_EXPAND_ENABLED` | `0` | 低置信度自适应查询扩展（M3.4）：信号不足时生成最多 2~3 个改写再补救检索一次；默认关闭 |
| `QUERY_EXPAND_MODEL` | 同 `QUERY_REWRITE_MODEL` | 扩展变体生成的模型 |
| `QUERY_EXPAND_MAX_VARIANTS` | `3` | 单次补救最多生成的改写变体数 |
| `CHAPTER_EXPANSION_MODE` | `off` | 证据带入方式：`off/neighbors` 用现有邻居机制 / `chapter` 命中片段所在整章按原文顺序进入 prompt |
| `CHAPTER_EXPANSION_MAX_TOKENS` | `3000` | 整章扩展的真实 token 预算闸门，超预算从离命中最远处截断并写 trace |
| `GRAPH_ENABLED` | `0` | 设成 `1` 建人物关系图，让"某某有哪些伴侣/师父"这类问题能查图而不是靠碰运气 |
| `GRAPH_MAX_CHUNKS_PER_RELATION` | `80` | 成本闸门：每个「书×关系」最多采样多少片段去抽人名 |
| `GRAPH_MODEL` | `glm:glm-4-flash` | 抽人名用的模型（便宜的小模型就够） |
| `MAX_UPLOAD_BYTES` | `20971520` | 单个上传文件的大小上限（字节），按 1MB 分块流式读取，超限返回 413 |
| `LOG_LEVEL` | `INFO` | 后端日志级别（DEBUG/INFO/WARNING/ERROR） |
| `DB_POOL_MIN_SIZE` | `1` | PostgreSQL 连接池最小连接数（只有 FastAPI 后端会用到） |
| `DB_POOL_MAX_SIZE` | `10` | PostgreSQL 连接池最大连接数 |
| `ZHIPU_API_KEY` | 空 | 智谱开放平台 API Key，设置后界面出现 GLM 模型分组 |
| `ZHIPU_API_URL` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | 智谱 chat completions 接口地址 |

例如临时换用更小的模型作为默认值启动后端：

```bash
OLLAMA_MODEL=qwen2.5:3b uvicorn backend.main:app --port 8000
```

## 已知限制（demo 阶段）

- 仅支持 `.txt` 纯文本，如果是 epub/pdf 需要先自行转换成 txt。
- 章节识别覆盖常见“第 N 章/卷、序章、番外”等标题；非常规排版仍可能识别不到，此时章节字段为空但片段仍可检索。
- 后台任务状态会落库（`index_task_runs` 表）：刷新网页可以恢复进度，重启后端后侧栏也能恢复上一次任务的最后已知状态；上次进程遗留的进行中任务会被如实标记为 failed 并提示重试。数据库按书事务和哈希清单仍会保留，重新点击同步即可从未完成的书继续。
- 网页上传没有做用户隔离和内容审核，仅适合本地单人使用；如果要对外提供服务，需要额外加上大小限制、格式校验和按用户隔离存储。
- **文本编码**：`loader.py` 会依次尝试 UTF-8 → GB18030 解码，覆盖国内小说站常见的两种编码。如果上传后回答明显文不对题（比如问主角是谁答非所问），先检查文件真实编码（`file -I data/novels/xxx.txt`），极少数生僻编码可能仍需手动转码为 UTF-8 后再上传。
- **小模型幻觉**：`qwen2.5:3b` 参数量较小，即使检索到了正确的原文片段，也可能忽略上下文、凭训练时的记忆编造答案（例如把小说人物错认成其他知名小说的角色）。真实使用建议用 `qwen2.5:7b` 或更大的模型，指令遵循和上下文依据能力明显更强。
- **层级摘要是导航摘要**：当前采用抽取式采样，不等于完整的文学主题总结。它能把全局问题分散到多个相关章节，但深层主题仍取决于原文覆盖和生成模型判断。
- **Agent Lab 不做持久恢复**：轨迹只属于当前请求；刷新后不会从中间工具步骤继续。需要检查点和人工审批时再考虑 LangGraph。

## 版权与许可

- **代码**：MIT 许可，见 [LICENSE](LICENSE)。
- **小说文本不随仓库分发**。`data/novels/` 已被 `.gitignore` 排除，只保留原创的示例短篇
  `雾隐山庄.txt` 用于跑通流程。你需要自备合法获得的小说文本，放进 `data/novels/` 后同步索引。
- **不要把版权原文提交进版本库**，也包括测试结果文件：`tests/run_qa_tests.py` 只落盘
  80 字摘录 + `novel`/`chunk_id` 定位信息（早期版本会把整段原文写进结果 JSON，
  一次测试就写入十几万字，已修正）。需要核对完整原文时按 `novel` + `chunk_id` 去库里查。
- 本项目仅供个人学习与本地使用，请勿用于分发或提供公开的小说内容服务。
