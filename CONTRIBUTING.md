# 参与贡献

这是一个本地小说 RAG 问答 demo。欢迎修 bug、提改进；下面是开发和提交的约定。

## 开发环境

完整的环境搭建（Ollama、PostgreSQL + pgvector、Python venv、前端依赖）见
[README.md](README.md) 的「环境准备」。跑起来大致是：

```bash
# 后端（端口 8000）
source venv/bin/activate
uvicorn backend.main:app --port 8000

# 前端（端口 5173，另开一个终端）
cd frontend && npm run dev
```

## 分支与合并流程

`main` 分支已开启保护：禁止 force push、禁止删除，并且必须通过 PR 和 CI 合并。
不要直接执行 `git push origin main`；新提交还没有 required status checks，直推会被保护
规则拒绝。

标准流程如下：

```bash
git switch -c feat/xxx
# 修改代码，并按 Conventional Commits 提交
git push -u origin feat/xxx
gh pr create --fill
# 等待 GitHub Actions 全部通过
gh pr merge --squash --delete-branch
```

分支名建议与提交类型和功能对应。使用 squash 合并，保持 `main` 历史简洁。

## 提交前自查

CI 会在每个 PR 上跑这三项，本地先过一遍能省一个来回：

```bash
# 前端类型检查
cd frontend && npx tsc --noEmit

# 前端 e2e 测试（首次要装一次浏览器内核）
cd frontend && npx playwright install --with-deps chromium   # 只需装一次
cd frontend && npx playwright test

# 后端导入（能抓语法错误、坏 import）
python -c "import backend.main"

# 后端单元测试
python -m pytest
```

e2e 测试在 `frontend/e2e/`，所有 `/api/*` 请求都在 `e2e/mock-api.ts` 里被拦截、
返回固定假数据——**不需要真实后端、PostgreSQL、Ollama 或任何云端 key**，也不会消耗
任何账号额度。改到 `App.tsx` / `MessageBubble.tsx` 这类前端交互逻辑时，顺手加一条
对应的 e2e 用例；只改样式/文案一般不需要。

`tests/backend/` 下是后端的 pytest 单元测试，同样全部用 mock/monkeypatch，
不需要真实数据库、Ollama 或云端 key。改到 `backend/` 或 `src/` 下的逻辑时，
顺手补一条对应用例。

> `@playwright/test` 版本锁定在 `1.50.0`（`package.json` 里不带 `^`），因为更新的版本
> 要求 Node 20+，而本项目文档写的是 Node 18+。升级前先确认 Node 版本要求是否放宽。

改到检索/切分逻辑时，**先跑检索评测**（不需要起后端，但需要 PostgreSQL 索引已建好）：

```bash
# 改之前存一份基线
python scripts/eval_retrieval.py --save /tmp/before.json
# ...改代码...
# 改之后对比，逐条看哪些用例变好、哪些变差
python scripts/eval_retrieval.py --compare /tmp/before.json
```

这个脚本算的是 Recall@k 和 MRR，能客观判断改动是真的变好还是只是换了一批失败
案例——原理和指标定义见 [docs/rag-techniques.md](docs/rag-techniques.md)。
`tests/eval_baselines/` 下存着每个改进阶段的历史基线，可以直接拿来对比。

想连生成质量一起看（需要先起后端）：

```bash
python tests/run_qa_tests.py --model qwen2.5:7b --out tests/results_7b.json
```

改到前端流式渲染时，参考 [tests/PERF_CHECK.md](tests/PERF_CHECK.md) 确认无卡顿。

## CI 构建通知（飞书，可选）

每次 CI 跑完（不管成功还是失败）都会尝试把结果推到飞书群，由 `.github/workflows/ci.yml`
里的 `notify` job 调用独立的
[`ningkaikok/feishu-notify-action`](https://github.com/ningkaikok/feishu-notify-action)
实现（通用逻辑抽成了 Action，方便其他项目复用）。**默认不配置也没事**——
没配 webhook 地址时会直接跳过，不会让 CI 变红。

配置步骤：

1. 飞书群「设置」→「群机器人」→「添加机器人」→「自定义机器人」，拿到 Webhook 地址。
   建议同时勾选「签名校验」，多要一个签名密钥，防止别人拿着地址冒充机器人发消息。
2. **不要把这两个值贴在 issue、PR、聊天记录里**——跟 API key 一样，一旦出现在任何
   聊天/日志里就该视为已泄露，去飞书机器人设置里重置。在自己终端里执行：

   ```bash
   gh secret set FEISHU_WEBHOOK_URL       # 粘贴 Webhook 地址，回车确认
   gh secret set FEISHU_WEBHOOK_SECRET    # 如果开了签名校验，同样设置
   ```

3. 下一次 push 或 PR 触发 CI 时就会收到一张卡片消息：全部通过是绿色标题，
   有检查没过是红色；内容包含分支/PR、提交、触发人，Push 触发时带上这次提交信息，
   PR 触发时带上 PR 标题和链接，底部有个「查看详情」按钮直接跳到这次运行。

`notify` job 没有被列入 main 分支保护的必需检查（见上方「分支与合并流程」），
所以就算飞书配置错了（比如密钥填错、机器人被移出群），也只会让这一个 job 显示失败，
不会挡住任何 PR 合并。

## 提交规范

本项目使用 [Conventional Commits](https://www.conventionalcommits.org/)，格式为：

```text
<type>(<scope>): <description>
```

允许的 type：

| type | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | 修 bug |
| `perf` | 性能优化 |
| `refactor` | 不改变外部行为的重构 |
| `docs` | 文档 |
| `test` | 测试 |
| `chore` | 构建、依赖和杂项 |

本项目常用 scope：`retrieval`、`ingest`、`ui`、`backend`、`models`、`deps`。
description 可以使用中文，但 type 和 scope 必须使用英文小写。

示例：

```text
feat(ingest): 支持单书增量索引
fix(api): 处理任务取消时的竞态条件
```

## CHANGELOG 规范

`CHANGELOG.md` 使用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式。
只有以下提交类型需要进入 CHANGELOG：

| 提交 type | CHANGELOG 分组 |
| --- | --- |
| `feat` | `### Added` |
| `fix` | `### Fixed` |
| `perf`、`refactor` | `### Changed` |

`docs`、`style`、`chore`、`test` 不收录。条目必须从用户视角描述收益或修复结果，
不要只复述内部实现。新条目放在 `## [未发布]` 下；发布时填写版本号和绝对日期。

配置位于 `cliff.toml`。日常生成未发布内容时，先写入临时文件，再合并未发布段：

```bash
git cliff --unreleased --strip header --output /tmp/unreleased.md \
  && python3 scripts/merge_changelog.py /tmp/unreleased.md CHANGELOG.md
```

不要使用 `git cliff --prepend`，它会重复插入标题并破坏手写说明。生成后仍需人工润色。

发布版本时可以运行：

```bash
git cliff --tag v0.2.0 -o CHANGELOG.md
```

本项目不使用 CI 自动提交 CHANGELOG：默认 `GITHUB_TOKEN` 创建的 PR 不会触发新的
workflow，而 `main` 的 required checks 又要求 CI 结果，自动 PR 会因此无法合并。对当前
项目体量，手动生成更简单，也无需额外维护 PAT 或 GitHub App 凭证。

2026-08-01 及之前的旧提交不符合 Conventional Commits，无法由 `git-cliff` 自动收录；
对应历史 CHANGELOG 保持手写内容。

## 版权红线

- **不要提交任何版权小说原文**，包括测试结果文件（`tests/run_qa_tests.py` 已改成只存
  80 字摘录 + 定位信息）。`data/novels/` 除原创示例 `雾隐山庄.txt` 外都被 `.gitignore` 排除。
- **不要提交密钥**。密钥放项目根 `.env`（已被忽略），提交前 `git status` 确认没有 `.env`
  或含 key 的文件在待提交列表里。详见 [SECURITY.md](SECURITY.md)。
