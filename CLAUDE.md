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

### 自动生成

配置在 [cliff.toml](cliff.toml)，用 [git-cliff](https://git-cliff.org/) 从提交记录生成：

```bash
brew install git-cliff                            # 装一次
git cliff --unreleased --prepend CHANGELOG.md      # 把新提交插到「未发布」段
git cliff --tag v0.1.0 -o CHANGELOG.md             # 发版：把未发布定为 v0.1.0
```

映射关系、分组顺序、以及排除 `docs`/`chore`/`test` 都已在 `cliff.toml` 里配好，
不符合 Conventional Commits 的提交会被跳过并给出 WARN。

**自动生成只是起点，不是终点。** 它直接把 commit 的 description 抄进 CHANGELOG，
所以「从用户视角写」这条规范实际上要在**写提交信息时**就做到；生成后仍需过一遍，
把太技术的表述改成用户能看懂的话。

> 注意：本项目 2026-08-01 及之前的提交是中文自然语言、不符合 Conventional Commits，
> 无法被自动收录。那段 CHANGELOG 是手写的，保留原样；从下一个提交起启用自动生成。
