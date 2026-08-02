# 项目约定

## 分支与合并流程

`main` 分支已开启保护（2026-08-01 起）：禁止 force push、禁止删除，且**必须走 PR
并通过 CI 才能合并**——`git push origin main` 直推会被拒绝，这不是配置错误，是
GitHub 的既定行为：required status checks 只在 PR 合并时生效，因为一个刚创建、
从未跑过 CI 的 commit 天然没有"检查通过"的记录，直推必然被拒。

因此改动流程固定为：

```bash
git checkout -b feat/xxx                    # 从 main 切新分支，分支名建议对应 commit 的 type/scope
# ...改代码、按 Conventional Commits 提交...
git push -u origin feat/xxx
gh pr create --fill                         # 或写清楚标题正文
# 等 GitHub Actions 的两个检查（前端类型检查、后端导入检查）跑绿
gh pr merge --squash --delete-branch        # 通过后合并，squash 保持 main 历史整洁
```

**不要试图先推到 main 探路**——会被 protected branch hook 拒绝，属于预期行为，
不用怀疑是不是权限或网络问题，直接切分支走 PR 即可。

## Git 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/)。

格式：

```
<type>(<scope>): <description>
```

允许的 type：

| type | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | 修 bug |
| `perf` | 性能优化 |
| `refactor` | 重构（不改变外部行为） |
| `docs` | 文档 |
| `test` | 测试 |
| `chore` | 构建、依赖、杂项 |

示例：

```
feat(auth): add login page
fix(api): handle timeout error
```

本项目常用 scope：`retrieval`（检索）、`ingest`（切分入库）、`ui`（前端界面）、
`backend`（FastAPI）、`models`（生成模型接入）、`deps`。

**description 用中文写没问题**，但 `<type>(<scope>):` 前缀必须是英文小写，
否则自动生成 changelog 时无法归类。

## CHANGELOG 规范

用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式维护 `CHANGELOG.md`。

只收录这三类，按下表映射到分组：

| 提交 type | CHANGELOG 分组 |
| --- | --- |
| `feat` | `### Added` |
| `fix` | `### Fixed` |
| `perf`、`refactor` | `### Changed` |

**不收录**：`docs`、`style`、`chore`、`test`。

**从用户视角写条目**——写这个改动让用户能做什么、遇到的问题是否消失，
而不是复述代码怎么改的。

```
✅ 书名打错字也能查到，例如把"诡秘之主"打成"闺蜜之主"
❌ 在 _mentions_novel 里加入 Levenshtein 距离匹配
```

新条目放在 `## [未发布]` 下面；发版时把 `## [未发布]` 改成
`## [x.y.z] - YYYY-MM-DD` 并在上方新建空的 `## [未发布]`。

日期用绝对日期（`2026-08-01`），不要写"今天""上周"。

### 生成命令（手动跑，两条）

配置在 [cliff.toml](cliff.toml)，用 [git-cliff](https://git-cliff.org/) 从提交记录生成。
**不要用 `--prepend` 直接改 CHANGELOG.md**——它会往文件最前面硬插一整块（含重复的
「# 更新日志」标题），把手写说明挤下去。改用「生成到临时文件 + 只替换未发布段」：

```bash
brew install git-cliff    # 装一次

# 日常：把新提交整理进「未发布」段（手写说明和历史版本不会被动）
git cliff --unreleased --strip header --output /tmp/unreleased.md \
  && python3 scripts/merge_changelog.py /tmp/unreleased.md CHANGELOG.md

# 发版：把「未发布」定为具体版本号
git cliff --tag v0.2.0 -o CHANGELOG.md
```

映射关系、分组顺序、以及排除 `docs`/`chore`/`test` 都已在 `cliff.toml` 里配好，
不符合 Conventional Commits 的提交会被跳过并给出 WARN。

**生成只是起点，不是终点。** 它直接把 commit 的 description 抄进 CHANGELOG，
所以「从用户视角写」这条规范实际上要在**写提交信息时**就做到；生成后仍需过一遍，
把太技术的表述改成用户能看懂的话。

### 为什么没做成 CI 自动化

试过了，`.github/workflows/changelog.yml` 已因此删除（见 PR #8～#13 那段折腾）。
根因是 GitHub 的防递归设计：**用默认 `GITHUB_TOKEN` 创建的 PR 不会触发任何新的
workflow run**（`push` 和 `pull_request` 都不触发）。于是 bot 自动开的 CHANGELOG PR
上一个检查都没有，而 `main` 分支保护要求必需检查通过 → 永久卡在 BLOCKED 合不进去。

想让它真自动，只有这几条路，代价都不小：
- 建专用 PAT 或 GitHub App token 存 secret（多一个长期凭证要管）；
- 每次手动 Close 再 Reopen 那个 PR 来骗过限制（还是手工操作）；
- 关掉 `enforce_admins` 用管理员强合（削弱分支保护，不划算）。

对这个体量的项目，上面那条手动命令 1 秒钟就跑完，不值得为此加凭证或削弱保护。

> 注意：本项目 2026-08-01 及之前的提交是中文自然语言、不符合 Conventional Commits，
> 无法被自动收录。那段 CHANGELOG 是手写的，保留原样。
