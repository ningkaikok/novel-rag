# 小说 RAG 查询系统

[![CI](https://img.shields.io/github/actions/workflow/status/ningkaikok/novel-rag/ci.yml?branch=main&label=CI&logo=github)](https://github.com/ningkaikok/novel-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ningkaikok/novel-rag?label=release&color=orange)](https://github.com/ningkaikok/novel-rag/releases/latest)
[![License](https://img.shields.io/github/license/ningkaikok/novel-rag?label=License&color=blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](requirements.txt)
[![React](https://img.shields.io/badge/React-18-20232a?logo=react&logoColor=61DAFB)](frontend/package.json)

基于本地向量检索的小说问答 demo。检索和 Embedding 全部在本机运行；生成模型默认用本地 Ollama（不需要任何外部 API Key），也可以按需切换到你自己的 Claude 订阅（见下文"切换生成模型"）。

技术栈：
- **文本切分**：按段落聚合成约 500 字的片段，片段间有重叠，避免切断语义。
- **Embedding**：`sentence-transformers`，本地模型 `BAAI/bge-small-zh-v1.5`（中文效果较好，体积约 95MB）。
- **数据库**：PostgreSQL + pgvector，片段、原文和 embedding 存储在 `novel_rag` 数据库的 `novel_chunks` 表中。
- **生成模型**：默认本地 [Ollama](https://ollama.com)（`qwen2.5:7b`），界面里可随时切换到其他本地模型，或切到你自己的 Claude 订阅（云端，需额外知情——见下文）。
- **后端**：FastAPI（`backend/main.py`），把检索/生成逻辑包成 HTTP 接口，回答用 SSE 逐字流式返回。
- **前端**：React + Vite + TypeScript + Ant Design（`frontend/`），书卷气界面「书虫」。用 antd 组件 + `ConfigProvider` 主题令牌保留藏青主色与暖底书卷气，支持浅色/深色主题、上传/删除书籍、示例问题、流式回答，以及可点击的原文出处引用。
- **对话体验**：生成中可以点「停止」——不只是前端不再显示新字，后端会真的停止向模型索取内容（用云端模型时不多花钱）；刷新页面或重开浏览器后，之前的问答、原文出处、思考过程会自动恢复，中途被停止的那轮也会如实标出来；往上翻看历史时，下面来了新回答会提示「有新回复」，不会悄悄错过。

## 目录结构

```
novel-rag/
├── src/           # 核心业务逻辑（config/loader/ingest/rag），前后端共享
├── backend/       # FastAPI 后端（main.py + claude_cli.py + zhipu.py）
├── frontend/      # React + Vite 前端
├── tests/         # 问答测试集与脚本
├── data/novels/   # 放小说 .txt 文件
└── chroma_db/     # 旧 Chroma 数据目录（迁移后不再使用）
```

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

3. 创建 Python 虚拟环境并安装依赖（**注意：需要 Python 3.11/3.12/3.13**，3.14 太新，`tokenizers` 等库还没有对应的预编译包）：

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. 安装前端依赖（需要 Node.js 18+）：

   ```bash
   cd frontend
   npm install
   cd ..
   ```

## 添加你自己的小说

有两种方式，任选其一：

- **网页上传（推荐）**：启动网页后，在左侧「小说库」面板拖拽或选择 `.txt` 文件上传，再点「🔄 重建索引」即可生效。
- **手动放置**：把 `.txt` 文件放进 `data/novels/` 目录，再运行 `python src/ingest.py`。

每个文件当作一部小说，文件名会作为来源标注。目录里已经有一篇原创的示例短篇小说 `雾隐山庄.txt`，用于快速验证整个流程，可以直接删掉换成你自己的小说文本。

## 启动（FastAPI + React，推荐）

需要开两个终端：

**终端 1 — 后端 API（端口 8000）：**

```bash
source venv/bin/activate
uvicorn backend.main:app --port 8000
```

**终端 2 — 前端 dev server（端口 5173）：**

```bash
cd frontend
npm run dev
```

浏览器打开 `http://localhost:5173`。左侧「我的书架」可上传/删除小说、在「更多设置」里重建索引和调整参考原文数量；底部输入框会同时召回关键词和语义相关片段，合并排序后交给模型生成回答。问答命中片段后会自动带入同一本书前后相邻片段，回答会逐字流式显示，展开出处卡片可核实依据。前端 `/api` 请求由 Vite 代理到后端 `127.0.0.1:8000`。

> 生产部署：`cd frontend && npm run build` 生成静态文件，再由 FastAPI 或 Nginx 托管，即可单端口对外。

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

## 单独重建索引（命令行）

如果你是手动往 `data/novels/` 放的文件，或想在命令行重建：

```bash
source venv/bin/activate
python src/ingest.py
```

这一步会清空并重建 PostgreSQL `novel_chunks` 表里的向量索引；不会删除 `data/novels/` 原文，也不会删除旧的 `chroma_db/` 目录。

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
| `TOP_K` | `5` | 每次检索的片段数量 |
| `RECALL_K` | `20` | 关键词和向量检索各自召回的候选片段数量 |
| `CONTEXT_NEIGHBORS` | `1` | 问答时每个命中片段前后额外带入的相邻片段数 |
| `BM25_K1` | `1.2` | BM25 词频饱和速度，越小饱和越快（一般不用动） |
| `BM25_B` | `0.75` | BM25 文档长度归一化强度，0=不归一化、1=完全按长度惩罚（一般不用动） |
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

- 每次重建索引都是整体重建，暂不支持增量更新单本小说。
- 仅支持 `.txt` 纯文本，如果是 epub/pdf 需要先自行转换成 txt。
- 切分策略是通用的段落聚合，没有针对章节标题做特殊识别。
- 网页上传没有做用户隔离和内容审核，仅适合本地单人使用；如果要对外提供服务，需要额外加上大小限制、格式校验和按用户隔离存储。
- **文本编码**：`loader.py` 会依次尝试 UTF-8 → GB18030 解码，覆盖国内小说站常见的两种编码。如果上传后回答明显文不对题（比如问主角是谁答非所问），先检查文件真实编码（`file -I data/novels/xxx.txt`），极少数生僻编码可能仍需手动转码为 UTF-8 后再上传。
- **小模型幻觉**：`qwen2.5:3b` 参数量较小，即使检索到了正确的原文片段，也可能忽略上下文、凭训练时的记忆编造答案（例如把小说人物错认成其他知名小说的角色）。真实使用建议用 `qwen2.5:7b` 或更大的模型，指令遵循和上下文依据能力明显更强。

## 版权与许可

- **代码**：MIT 许可，见 [LICENSE](LICENSE)。
- **小说文本不随仓库分发**。`data/novels/` 已被 `.gitignore` 排除，只保留原创的示例短篇
  `雾隐山庄.txt` 用于跑通流程。你需要自备合法获得的小说文本，放进 `data/novels/` 后重建索引。
- **不要把版权原文提交进版本库**，也包括测试结果文件：`tests/run_qa_tests.py` 只落盘
  80 字摘录 + `novel`/`chunk_id` 定位信息（早期版本会把整段原文写进结果 JSON，
  一次测试就写入十几万字，已修正）。需要核对完整原文时按 `novel` + `chunk_id` 去库里查。
- 本项目仅供个人学习与本地使用，请勿用于分发或提供公开的小说内容服务。
