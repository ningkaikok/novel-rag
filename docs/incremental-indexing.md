# 增量索引与后台任务

M2 解决的是旧索引链路的两个结构性问题：增加一本书也要重新计算另外三本；而且
`DROP TABLE` 发生在耗时工作完成之前，一旦进程、数据库或模型出错，整个书架可能
暂时没有可用索引。

## 一、怎样判断一本书是否变化

PostgreSQL 新增 `index_manifest`：

```sql
CREATE TABLE index_manifest (
    novel         TEXT PRIMARY KEY,
    source_hash   TEXT NOT NULL,
    pipeline_hash TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL,
    indexed_at    TIMESTAMPTZ NOT NULL
);
```

- `source_hash` 是 txt 原始字节的 SHA-256。正文、编码或换行变化都会触发更新。
- `pipeline_hash` 包含切分大小、重叠、Embedding 模型、上下文增强和关系图配置。
  文件不变但索引算法变化时也会正确失效。
- 当前目录有、清单没有的是新增；哈希不同的是修改；数据库有、目录没有的是删除；
  两个哈希都一致的书直接跳过。

M2 之前的数据库没有 manifest。第一次运行会把已有书分类为“修改”，逐本建立清单；
这是一次性迁移。之后无变化的同步只做文件哈希和少量 SQL 查询。

## 二、为什么取消不会留下半套索引

每本变化书分成“准备”和“切换”两段：

```text
事务外：章节切分 → Context（可选）→ 分批 Embedding → BM25 分词 → 关系边（可选）
                                           ↓ 全部成功后
事务内：删除该书旧 BM25/向量/关系 → 写新向量 → COPY 新 BM25 → 更新 manifest → COMMIT
```

耗时计算期间旧版仍能被查询。最后的 `replace_novel_index()` 只使用一个 PostgreSQL
事务：写入失败、用户取消或进程退出都会回滚当前书，读取方只会看到完整旧版或完整
新版。已经完成的其他书不回滚，因为它们各自已经是一个一致的检查点；重试时 manifest
会跳过它们，只继续未完成内容。

这比“先删旧数据再慢慢算”多占一点临时内存，但把可用性和一致性边界变得非常清楚。

## 三、后台任务 API

上传和删除接口保存文件后立即返回任务，不再占住请求几分钟：

| 接口 | 作用 |
| --- | --- |
| `POST /api/books` | 原子保存 txt，启动增量同步 |
| `DELETE /api/books/{name}` | 删除文件，启动索引清理 |
| `POST /api/reindex?force=false` | 扫描变化并同步；`force=true` 才重做全部 |
| `GET /api/index-tasks/current` | 页面刷新后恢复最近任务 |
| `GET /api/index-tasks/{id}` | 查询阶段、百分比、结果或失败原因 |
| `POST /api/index-tasks/{id}/cancel` | 发出安全停止信号 |
| `POST /api/index-tasks/{id}/retry` | 重新扫描，只继续仍未完成的书 |

当前项目是本地单用户应用，所以只允许一个索引任务同时运行，使用一个后台线程，
不额外引入 Redis/Celery。任务卡状态存进程内存；真正需要跨重启保存的数据边界是
`index_manifest` 和单书事务。后端重启导致任务卡消失时，再点一次同步即可恢复工作。

## 四、取消的响应边界

Embedding 改为每 32 段一批，每批之间检查取消信号；BM25 每 25 段检查，COPY 入库
每 50 段检查。因此默认配置下停止通常很快，但不会粗暴杀线程。若开启 Contextual
Retrieval，正在执行的单次外部模型调用不能从 Python 线程安全强杀，会在当前调用返回
后停止；这比留下未知状态更可靠。

## 五、验证方法

```bash
# 第一次：迁移旧索引或处理真实变化
python src/ingest.py

# 第二次：应直接显示“所有小说都已是最新版本”
python src/ingest.py

python -m pytest
cd frontend && npx tsc --noEmit && npm run test:e2e
```

单元测试覆盖变化分类、只处理修改书、写库前取消、事务 COPY 失败、任务并发/取消/失败；
端到端测试覆盖刷新恢复进度、安全停止、失败原因和重试按钮。
