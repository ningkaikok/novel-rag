# 生产部署与数据运维（M5 / M6.7）

## 部署边界

生产入口为 `backend.production:app`。每位用户对应一个租户，拥有独立的 worker
进程、PostgreSQL/pgvector 服务、小说/通用文档目录、模型缓存、聊天记录、检索缓存及
索引任务。Gateway 根据经过认证的身份选择固定 worker；客户端传入的租户、用户、
worker 密钥及转发身份头都会被丢弃。数据库和 worker 不发布宿主机端口。

采用独立实例，是因为现有后端的模型选择、RAG、缓存及任务管理器是进程级状态。
用户不能指定另一个租户的 worker，也不能通过猜测 session/run/document/job ID 跨租户
读取数据。租户网络彼此分开，Gateway 和运维人员是受信任的管理边界。

当前不提供同一租户内多人共享书架或成员角色；新增用户应创建新租户。相比共享数据库
行级隔离，这种方式每位用户需要额外数据库与模型内存，适合小规模私有部署。

`backend.main:app` 与根目录 `docker-compose.yml` 保留本地单用户运行方式。
它们没有用户鉴权，不能作为生产入口发布到公网。CLI/MCP 同样属于操作者本地权限，
不对普通租户用户开放。

## 初始化和启动

```bash
python scripts/production_init.py --users alice bob
docker compose --env-file deploy/private/.env -f deploy/private/compose.json up --build -d
```

初始化只允许写入空目录，不覆盖已有身份或数据。生成文件全部位于被 Git 和 Docker
build context 排除的 `deploy/private/`。默认数据目录为空，不复制本地已有小说或聊天。

`deploy/private/.env` 保存管理 token、各用户的随机 token 和 worker 密钥；
`alice.env` 等保存对应数据库凭据及 worker 设置。`tenants.json` 只保存用户 token
的 SHA-256 摘要和配额配置。随机 token 有 384 位熵，不支持弱密码。

浏览器打开 `http://localhost:8000`，HTTP Basic 登录用户名为租户 ID、密码为该用户
token；API 客户端使用 `Authorization: Bearer <用户 token>`。凭据不能放 URL。
对外服务时用 HTTPS 反向代理连接回环端口；反向代理保持 Host，不缓冲 SSE。
入口拒绝跨站写请求，认证响应及用户数据不缓存。

不要对外开放 worker、PostgreSQL、Prometheus 或 Alertmanager 端口。不要把私有目录、
`docker compose config` 的完整输出、数据库 dump 或源文件归档放进 PR/日志/截图。
Linux 生成文件权限为 0600、根目录为 0700；Windows 应在目录 ACL 中只授权操作者。

## 配额与超限处理

每个 `tenants.json` 条目可配置以下字段，修改后重启 Gateway：

| 字段 | 默认值 | 口径 |
| --- | --- | --- |
| `calls_per_day` | 200 | UTC 自然日内通过准入的 POST 操作次数，包括问答、核验、上传、重建、模型切换 |
| `requests_per_minute` | 120 | UTC 固定分钟内进入 worker 的 HTTP 请求数 |
| `max_concurrent` | 2 | 直到响应完成或断开才释放的请求槽，包含 SSE |
| `storage_bytes` | 1 GiB | 源文件目录中的总字节数，包括待替换临时文件 |
| `request_timeout` | 1800 秒 | 单次请求总时长上限 |
| `enabled` | true | false 时拒绝该用户的所有访问 |

SQLite `control/quotas.sqlite` 以事务实现准入和上传空间预留，多个 Gateway 进程共享
同一配额库；重启不会重置每日用量。上传需要 Content-Length，按请求大小的两倍预留
空间，覆盖 multipart 和潜在兼容副本。上传上限仍受单文件、PDF 页数及解析片段数限制。
不满足配额时返回 429 或存储超限 413，不开始 worker 操作。

失败、客户端取消及 worker 故障也会消耗已准入的 POST 配额，以避免反复失败绕过限额；
请求槽及预留空间始终释放，进程异常退出的租约在硬超时之后回收。每日额度在 UTC
日期切换后自动重置。`GET /api/usage` 返回当前用户的用量，不返回其他租户信息。

存储配额计量上传源文件，不是数据库物理空间；向量/BM25/历史派生数据由保留策略与
数据库容量监控管理。调用配额计量 API 操作，不是供应商账单 token；已有 Model Gateway
的输出/估算成本预算继续生效，真实供应商用量对账仍需后续接入。

## 数据保留和删除

维护操作先持久化暂停该租户入口、排空 HTTP/SSE，再停止其 worker。停止 worker 会
结束索引和引用核验后台线程，避免清理后再次写入。其他租户继续服务。维护失败保持
暂停；必须检查原因后显式 `resume`。

预览最近 30 天之外的聊天、摘要、引用反馈/核验、运行事件和终态索引任务：

```bash
python scripts/production_ops.py retention --tenant alice --days 30
```

清理历史，同时删除 180 天之前未更新的文档全部版本及源文件：

```bash
python scripts/production_ops.py retention --tenant alice --days 30 --document-days 180 --apply --confirm-tenant alice
python scripts/production_ops.py resume --tenant alice
```

不指定 `--document-days` 时保留文档。V2 文档以最近版本时间判定，V1 小说和孤立源文件
以文件 mtime 判定；新版本仍在保留期内时保护该文档的来源文件。删除旧聊天会同时
使包含旧内容的摘要失效。V2 清理按外键依赖删除版本、片段、term 和 manifest；来源
文件在数据库提交后删除，失败可以重跑。活动索引任务不会作为过期历史清掉。

用户可以继续使用现有会话清空、文档删除 API。删除整个账号的数据：

```bash
python scripts/production_ops.py purge --tenant alice --confirm-tenant alice
```

此操作撤销用户访问、停止 worker、删除该租户数据库中的在线/恢复副本、源文件及
配额/告警记录及工具生成的恢复前源文件副本。其他用户不受影响。运维保存的外部备份
需要按相同策略销毁；不能用旧备份恢复已删除账号。数据库物理卷及宿主机快照的销毁由存储管理员
负责。保留清理工具可由部署主机的任务计划运行；当前仓库不替操作者创建定时任务。

Gateway 每小时清除 30 天之前的用量与已解决告警；Prometheus 默认保留 30 天。
容器日志应由部署主机配置轮转和最长保留期。

## 备份与恢复

```bash
python scripts/production_ops.py backup --tenant alice --backup backups/alice-2026-10-12
python scripts/production_ops.py verify --tenant alice --backup backups/alice-2026-10-12
python scripts/production_ops.py restore --tenant alice --backup backups/alice-2026-10-12 --confirm-tenant alice
```

备份为一个私有目录，包含 custom-format PostgreSQL dump、源文件 tar、该租户配额
快照和 SHA-256 manifest。完成后恢复服务；失败备份没有有效 manifest，租户保持暂停。
私有凭据不写入备份，异机恢复前先创建同 ID 的私有租户部署，另行保管和恢复服务配置。
校验和用于发现损坏；只恢复经过操作者审核的可信 dump，不能执行外部提供的 SQL 备份。

恢复先验证租户、完整校验和及归档路径，拒绝绝对路径、目录穿越、符号链接和特殊文件。
文件先解包到独立暂存目录；数据库先恢复到全新的临时数据库，再切换当前数据库，
不会残留备份中不存在的新表。旧数据库与源文件副本保留供回滚；源文件副本中的
`database.json` 记录对应旧数据库名。配额按日取已有用量和备份用量的较大值，恢复
不能降低当天配额，也不会修改其他用户的用量。

恢复后保持暂停，检查文档目录、版本、聊天与问答，再运行 `resume`。恢复过程发生
文件切换错误时保持暂停；使用保留的旧数据库和源文件副本回滚后再恢复服务。建议
每天备份、至少每次发布前备份，并定期在隔离部署演练恢复；备份目录按既定天数保留，
不要长期保留过期用户数据。

数据库一致性依据 [PostgreSQL dump 文档](https://www.postgresql.org/docs/16/backup-dump.html)；
数据库重命名限制见 [ALTER DATABASE](https://www.postgresql.org/docs/16/sql-alterdatabase.html)。

## 日志、监控与告警

Gateway 写 JSON 审计事件，包含 request_id、tenant_id、方法、状态和耗时；worker
设置 `LOG_FORMAT=json` 后记录代码位置和操作字段，省略旧日志中的问题、正文、路径、
密钥及异常文本。启动命令关闭 Uvicorn access log，避免 URL 中的查询内容进入日志。

`/metrics` 需要管理 token，仅暴露固定指标和租户 ID。Prometheus 使用私有 token 文件
抓取服务状态、数据库字节数、源存储、每日调用、错误、配额拒绝、索引失败及请求耗时。
备份工具记录最近成功备份时间。维护模式单独可见，计划维护时抑制 worker/database
不可用告警；暂停超过一小时仍告警。

生成的部署同时启动 Prometheus 和 Alertmanager（界面分别绑定本机 9090/9093）。
规则覆盖 Gateway/worker/数据库不可用、存储 85%、调用 90%、重复错误、配额拒绝、
索引失败、慢请求与超过 24 小时没有成功备份。Alertmanager 将触发/解决事件发给
经过管理认证的 `/admin/alerts`，Gateway 持久化去重状态。用户通过 `/api/alerts`
只看到本租户和全局基础设施告警。

可以在私有 `alertmanager.yml` 增加邮件或 Webhook 接收器并重启 Alertmanager；默认
闭环止于本机 UI 与租户 API，不发送外部消息。规则与接收器分别依据
[Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
和 [Alertmanager 配置](https://prometheus.io/docs/alerting/latest/configuration/)。

## 发布与验证

先备份，在一个租户上重建并启动 worker，确认指标、检索与流式问答后逐个租户发布。
每个私有 worker env 都可以独立配置现有 V2 `off/gray/on` 灰度开关；关闭开关并重启该
worker 即回滚检索灰度。需要回滚代码时恢复上一镜像，不覆盖用户数据。
Agent/MCP 仍只开放只读工具；数据运维只接受独立管理凭据和显式租户确认。

仓库包含认证/隔离、并发配额、断流释放、恶意归档、恢复保留、保留删除和隐私日志
mock 测试。真实验收需要 Docker 和 pgvector：启动两位用户，分别上传不同内容，
交叉请求相同会话/文档 ID；触发配额、暂停、备份、恢复、清理和告警后确认隔离。
本次本机没有 Docker，尚未完成真实容器启动、pgvector 入库和数据库恢复演练；不能
把 mock 测试当成已经部署上线。
