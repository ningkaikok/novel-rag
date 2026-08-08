# 代码导读：从零理解一个 RAG 系统

这份文档回答的是「**这份代码怎么读**」。如果你想理解的是「某个技术为什么这么做、
效果如何」，看 [rag-techniques.md](rag-techniques.md)——那份按技术点组织，
每节都有真实的失败案例和实测数据。

两份文档的关系：

| | 组织方式 | 适合什么时候看 |
| --- | --- | --- |
| **本文** | 按代码结构 | 第一次接触这个项目，想知道从哪读起 |
| [rag-techniques.md](rag-techniques.md) | 按技术点 | 想深入某个具体技术（BM25/重排/Contextual Retrieval…） |

---

## 一、先建立全局图景

### 一次提问的完整路径

```
用户在界面输入问题
    ↓  frontend/src/api.ts  askStream()
POST /api/ask                                    backend/main.py
    ↓
① 多轮改写      "他后来怎么样了" → "李化元后来怎么样了"
    ↓           src/query_rewriter.py
② 多路检索                                        src/rag.py
    ├── 语义检索      向量相似度，懂近义但对人名不可靠
    ├── BM25 检索     按词精确匹配，人名/专有名词的强项
    ├── 结构性检索    按位置取书的开头/结尾
    ├── RRF 融合      三路合并成候选池
    └── 交叉编码器重排 从候选池精选                src/reranker.py
    ↓
③ 补相邻片段    避免语义被切断
    ↓
④ 拼 prompt，按模型前缀路由                       backend/{zhipu,claude_cli}.py
    ↓
⑤ SSE 流式推送 trace / sources / token / done
    ↓
⑥ 落库到 chat_turns                              src/postgres.py
```

### 目录职责

```
src/          纯业务逻辑，不依赖 Web 框架
backend/      只做"把 src 包成 HTTP"这一件事
frontend/     React 界面
scripts/      评测等独立工具
docs/         本文档所在
tests/        pytest（后端）+ 评测集与历史基线
```

**为什么 `src/` 和 `backend/` 要分开**：这条边界让检索逻辑可以脱离 HTTP
单独验证——`scripts/eval_retrieval.py` 直接调 `src/rag.py`，完全不经过 FastAPI；
`python src/ingest.py` 也能独立跑。如果把检索逻辑写在路由函数里，
想测一下检索质量就得先起一个 Web 服务，非常别扭。

---

## 二、建议的阅读顺序

不要从 `main.py` 开始逐行读——那样会淹没在路由和流式细节里。按数据流动的顺序读：

### 第 1 站：`src/loader.py` — 文本怎么变成可检索的片段

**看点**：为什么不能整本书直接塞进向量、切分要平衡什么、重叠是干什么的。

这个文件的模块 docstring 里记录了一个真实的坑：早期版本按空行切分，切出平均
3108 字的巨块，而 embedding 模型上限是 512 token——**每块只有开头约 500 字
进了向量，后面 84% 的内容在检索中完全不可见**。文本在库里，却永远搜不到。

> 关键认知：超出 embedding 上限的部分**不会报错**，只是静默丢失。

### 第 2 站：`src/ingest.py` — 片段怎么变成两套索引

**看点**：为什么要建两套索引、它们分别擅长什么。

```
向量索引（HNSW）   按语义找，懂近义，但对"必须逐字匹配"的人名不可靠
BM25 倒排索引      按词精确匹配并加权，人名/专有名词的强项，但完全不懂近义
```

这两套**必须基于同一批文本同时重建**，否则检索结果会自相矛盾。

顺带看 `_build_contexts`：这是 Contextual Retrieval 的成本控制，三道闸门
（按书跳过 / 只处理缺上下文的片段 / 按内容哈希增量复用）。

### 第 3 站：`src/tokenizer.py` — 一个小而关键的模块

**看点**：为什么分词规则必须索引和查询共用。

只有 65 行，但它体现了一个重要原则：如果建索引时把「长春功」切成一个词、
查询时切成「长春」+「功」，两边的词表根本对不上，**BM25 会静默失效**。
所以分词规则只能有一份。

### 第 4 站：`src/rag.py` — 核心，检索的全部逻辑

这是最大的文件（约 700 行），建议按这个顺序读里面的函数：

| 函数 | 做什么 | 看点 |
| --- | --- | --- |
| `retrieve` | 向量检索 | pgvector 的 `<=>` 余弦距离操作符 |
| `keyword_retrieve` | **BM25** | 公式三项逐项对应写在 SQL 里，注释里拆解了每项解决什么问题 |
| `positional_retrieve` | 结构性检索 | 为什么"结局是什么"这类问题语义检索必然失败 |
| `retrieve_hybrid_traced` | 串起全流程 | RRF 融合、重排的接入点、trace 怎么生成 |
| `expand_neighbors` | 补相邻片段 | |
| `build_prompt` | 拼 prompt | |

**读 `keyword_retrieve` 时重点看** `_strip_novel_titles` 的说明：那里记录了一个
实测出来的 bug——用户问「《凡人修仙传》里韩立的绰号」时，书名被切成「凡人」
「修仙」当成内容词参与打分，给无关片段白送 14.1 分，把正确答案压了下去。
书名的职责是**路由**，不该再参与书内打分。

### 第 5 站：`src/reranker.py` — 检索系统最核心的架构权衡

**看点**：双编码器 vs 交叉编码器。

这个文件的模块 docstring 讲透了为什么真实系统要两阶段：

```
双编码器：问题和文档各自编码 → 能预计算 → 快，但精度有限
          （编码文档时根本不知道用户会问什么）
交叉编码器：两者拼一起送进模型 → 词能直接注意力交互 → 准，但没法预计算
          （全库跑一遍要几分钟）

所以：粗筛几万→几十（双编码器），精排几十→几个（交叉编码器）
```

### 第 6 站：`backend/main.py` — 怎么串成一个服务

现在再看这个文件就容易了。模块 docstring 里画了完整的请求流程图。

重点看 `/api/ask`：
- 为什么是 `async def` 而不是 `def`（关系到"用户点停止后还会不会继续烧钱"）
- `_next_or_sentinel` 这个哨兵模式（`StopIteration` 不能穿过 `await` 边界）
- SSE 事件序列怎么对应到界面上的三块区域

---

## 三、可以直接上手跑的实验

理解代码最快的方式是改一个参数、看指标怎么变。

### 实验 1：关掉 BM25，看纯语义检索能做到什么

```bash
python scripts/eval_retrieval.py --save /tmp/before.json
# 临时把 rag.py 里 RRF 融合的 keyword_sources 去掉，再跑
python scripts/eval_retrieval.py --compare /tmp/before.json
```

预期：问人名、专有名词的用例明显变差（`窝头`、`韩铸` 这类稀有词）。

### 实验 2：调 CHUNK_SIZE，看切分尺寸的影响

```bash
CHUNK_SIZE=200 python src/ingest.py    # 切得更碎
python scripts/eval_retrieval.py --compare tests/eval_baselines/02-contextual.json
```

⚠️ 会重建全库索引，约 3 分钟。

### 实验 3：看 Contextual Retrieval 到底改了什么

```bash
python -c "
import sys; sys.path.insert(0,'src')
from postgres import connect
with connect() as conn:
    for r in conn.execute('SELECT context FROM chunk_contexts LIMIT 5'):
        print(r['context'])
"
```

对照 `docs/rag-techniques.md` 第 3 节里的 A/B 数据，理解"给片段补一句话"
为什么能让检索从"完全找不到"变成"排第 5"。

### 实验 4：观察多轮改写

```bash
# 起后端后，连问两轮，看第二轮的 trace
tail -f /tmp/novel-rag-backend.log | grep 查询改写
```

---

## 四、这个项目里值得学的"工程判断"

技术实现之外，有几个决策过程比结论更值得看：

**① 先建评测，再谈优化**（[rag-techniques.md 第 0 节](rag-techniques.md#0-检索评测一切改进的前提)）

改检索最容易骗自己：试两个问题感觉"好像变好了"就收工。但检索是统计问题，
一个改动完全可能让 A 类变好、B 类变差。所以这一系列改造的第一步不是写功能，
而是写 `scripts/eval_retrieval.py`。

后来这个决定被反复验证是对的——BM25 那一轮的第一版实测**让某些用例变差了**
（Q1 从 #2 掉到 #14），没有评测根本发现不了。

**② 成本要实测，不能拍脑袋**（[第 3 节](rag-techniques.md#3-contextual-retrieval成本才是主要矛盾)）

Contextual Retrieval 看起来很美（官方数据检索失败率降 49%），但实测单次 LLM
调用 4.4 秒、全库要 40 小时。**成本控制不是可选项，是这个技术能不能用的前提**。

**③ 优雅降级不能吞掉失败原因**

这个教训在两个地方重复出现过。Contextual Retrieval 第一次跑完 451 条**全部失败**，
但日志只有一句"451 条生成失败"——真实原因（没加载 `.env`）完全看不出来。

降级是对的（不该让整个流程卡住），但**降级路径同样要可观测**。

**④ 边界情况会让启发式静默失效**

- 人物名提取的频次门槛写死为 20 → 只有 3 个片段的书里没有名字能达标 →
  名单成空集合 → 判据恒为真 → 所有片段都被误判
- 查询改写的长度阈值设成 12 → 正好 12 个字的自足问题被误判成追问

两个都是"代码没报错，功能静默走偏"。**写死的阈值要问一句：极端输入下它会怎样？**

---

## 五、还没做的部分（如实记录）

| 项目 | 状态 |
| --- | --- |
| 交叉编码器重排 | 代码完成、单元测试通过，但**模型在本机下载不了**，没有实测指标 |
| 长上下文取舍 | 四杠杆里唯一的空缺。`TOP_K=5` 是拍脑袋定的，没测过 3/5/10 哪个好 |
| GraphRAG | 未开始。能解决"韩立有哪些伴侣"这类全书范围的关系聚合问题 |
| 语义切分 / Late Chunking | 未做，需要换长上下文 embedding 模型 |

`CONTEXT_NEIGHBORS` 还有个已知现象：理论上该把 5 段扩展成 15 段，
实测扩展后还是 5 段（检索到的片段本来就相邻，去重后没增加）。待查。
