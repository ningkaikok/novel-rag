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
```

e2e 测试在 `frontend/e2e/`，所有 `/api/*` 请求都在 `e2e/mock-api.ts` 里被拦截、
返回固定假数据——**不需要真实后端、PostgreSQL、Ollama 或任何云端 key**，也不会消耗
任何账号额度。改到 `App.tsx` / `MessageBubble.tsx` 这类前端交互逻辑时，顺手加一条
对应的 e2e 用例；只改样式/文案一般不需要。

> `@playwright/test` 版本锁定在 `1.50.0`（`package.json` 里不带 `^`），因为更新的版本
> 要求 Node 20+，而本项目文档写的是 Node 18+。升级前先确认 Node 版本要求是否放宽。

改到检索/切分逻辑时，跑一遍问答自查（需要先起后端）：

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

3. 下一次 push 或 PR 触发 CI 时就会收到消息，格式类似：

   ```
   【ningkaikok/novel-rag】CI ❌ 有检查未通过
   分支/PR：main　提交：abcdef1　触发人：ningkaikok（push）
   ✅ 前端类型检查：success
   ✅ 前端 e2e 测试：success
   ❌ 后端导入检查：failure
   详情：https://github.com/ningkaikok/novel-rag/actions/runs/xxxxx
   ```

`notify` job 没有被列入 main 分支保护的必需检查（见下方「main 已开启保护」），
所以就算飞书配置错了（比如密钥填错、机器人被移出群），也只会让这一个 job 显示失败，
不会挡住任何 PR 合并。

## 提交规范

本项目用 [Conventional Commits](https://www.conventionalcommits.org/)，
具体格式、type 白名单、CHANGELOG 生成方式都写在 [CLAUDE.md](CLAUDE.md)。要点：

- 前缀 `<type>(<scope>):` 用英文小写，description 可以用中文。
- 只有 `feat` / `fix` / `perf` / `refactor` 会进 CHANGELOG。
- CHANGELOG 从用户视角写。生成命令见 [CLAUDE.md](CLAUDE.md)（用 `--output` 到临时文件
  再跑 `scripts/merge_changelog.py`，**不要用 `--prepend`**），生成后仍需润色。

## 版权红线

- **不要提交任何版权小说原文**，包括测试结果文件（`tests/run_qa_tests.py` 已改成只存
  80 字摘录 + 定位信息）。`data/novels/` 除原创示例 `雾隐山庄.txt` 外都被 `.gitignore` 排除。
- **不要提交密钥**。密钥放项目根 `.env`（已被忽略），提交前 `git status` 确认没有 `.env`
  或含 key 的文件在待提交列表里。详见 [SECURITY.md](SECURITY.md)。
