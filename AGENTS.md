# AI Agent 项目说明

本文件是各种 AI 编程 Agent 的统一入口。人类开发者和 Agent 都应以
[CONTRIBUTING.md](CONTRIBUTING.md) 为开发、测试、分支、提交和 CHANGELOG 规范的唯一来源。

## 开始工作前

- 阅读 [README.md](README.md) 了解项目结构、运行方式和环境变量。
- 阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，按改动范围执行对应检查。
- 查看 `git status`，保留用户已有的未提交修改，不覆盖无关文件。

## 项目约束

- 不提交 `.env`、API Key、Webhook、个人路径或本地服务凭据。
- 不提交版权小说原文；`data/novels/` 只允许仓库已有的原创演示数据。
- 修改 `backend/` 或 `src/` 时补充后端测试；修改前端交互时补充 E2E 测试。
- 修改检索或切分算法时，按贡献指南运行检索评测并检查指标回退。
- `main` 是受保护分支；使用独立分支、Conventional Commit、PR 和 squash merge。

工具专属且不适合共享的配置应放在各工具的本地配置文件中，不要写入本文件。
