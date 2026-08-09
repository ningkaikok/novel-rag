# 从本项目提炼的面试题

全部来自这个项目**真实踩过并验证过**的问题，不是教科书题目。每题给出：
考察什么、参考答案、加分点、以及对应的代码位置。

带 ⭐ 的是我认为最有区分度的题——多数候选人答不出细节。

**目录**

- [一、流式输出与中断](#一流式输出与中断)
- [二、前端渲染性能](#二前端渲染性能)
- [三、RAG 检索质量](#三rag-检索质量)
- [四、工程实践与 CI](#四工程实践与-ci)
- [五、测试设计](#五测试设计)

---

## 一、流式输出与中断

### Q1 ⭐ 用户点「停止」，你的后端会真的停下来吗？怎么验证？

**考察**：是否理解「中断」不只是前端的事；是否有验证意识而不是想当然。

**参考答案**

前端 `AbortController.abort()` 只切断了浏览器这一侧。如果后端不做任何处理，
生成任务会继续跑到底——**用云端模型时，用户按了停止你还在扣他的钱**。

完整链路需要：

```
abort() → 连接断开 → 后端 is_disconnected() 检测到 → 停止向上游索取 token
```

**验证方法**（关键在"证伪"）：提前断开连接，然后**隔一段时间再看内容有没有继续增长**。

```
curl --max-time 4 提前断开
断开瞬间：还没写库
12 秒后： 102 字 / interrupted   ← 不再增长
```

如果上游没停，GLM 会继续吐几千字。内容停住不动才能证明真的停了。

> 只看"前端不再显示新字"是**无效验证**——前端不显示不代表后端不生成。

**代码**：[`backend/main.py`](../backend/main.py) 的 `event_stream`

---

### Q2 ⭐ FastAPI 里 `def` 和 `async def` 的流式接口有什么区别？什么时候必须用 async？

**考察**：对 ASGI 事件循环的理解深度。这题能筛掉「只会写 CRUD」的候选人。

**参考答案**

同步 `def` 接口会被 FastAPI 丢进**线程池**执行。线程池里的任务：

- 拿不到 `request.is_disconnected()` 的及时结果（没有 await 点让出控制权）
- 客户端断开后**不会自动终止**，会一直跑到函数返回

所以「检测客户端断连并停止工作」这个需求**必须**用 `async def`——只有在协程里才能
`await` 出让控制权，让事件循环有机会处理连接状态变化。

**加分点**：主动指出「但生成器本身是同步的」这个矛盾（见下一题）。

---

### Q3 ⭐⭐ 上游 SDK 只给了同步生成器，你要在 async 接口里边流式转发边检测断连，怎么写？

**考察**：这是上一题的真实延伸，也是本项目里最容易写错的地方。

**参考答案**

直接在协程里 `for chunk in sync_generator` 会**阻塞事件循环**——整个 async 的意义
就没了，`is_disconnected()` 永远轮不到执行，断连检测静默失效（代码看起来对，
但功能不生效，最难查）。

正确做法：用 `run_in_threadpool` 逐个取，每次 `await` 都是一次让出控制权的机会。

```python
_SENTINEL = object()

def _next_or_sentinel(it):
    return next(it, _SENTINEL)

while True:
    chunk = await run_in_threadpool(_next_or_sentinel, token_iter)
    if chunk is _SENTINEL:
        break
    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
    if await request.is_disconnected():
        token_iter.close()   # 关掉生成器，上游 HTTP/subprocess 连接随之断开
        break
```

**为什么用哨兵而不是 `try/except StopIteration`**：`StopIteration` 不能穿过 `await`
边界，会被转成 `RuntimeError`。这是 PEP 479 的规定，很多人不知道。

**加分点**：说出 `token_iter.close()` 的作用——它触发生成器内部的 `GeneratorExit`，
让 `requests` 的流式连接或 `subprocess` 被关掉，这才是「真的停止索取」。

**代码**：[`backend/main.py`](../backend/main.py)

---

### Q4 ⭐ `AbortController.abort()` 之后，`fetch` 的 reader 一定会抛异常吗？

**考察**：是否真的动手调试过中断，而不是照抄教程。

**参考答案**

**不一定。** 这是我在本项目踩的真实坑：

第一版我只在 `catch`（`onError`）里处理中断标记，结果点了停止后**界面卡在生成中**
——光标还在转、停止键还在。查了才发现 `abort()` 后 `reader.read()`
**可能直接返回 `{done: true}` 而不抛 `AbortError`**，于是走的是正常结束路径。

修法：正常结束路径里也要判断「是否用户主动停止过」。

```ts
function finishTyping() {
  const stopped = userStoppedRef.current;   // 正常结束路径也要读这个标记
  patchLast(m => ({ ...m, streaming: false, interrupted: stopped || m.interrupted }));
}
```

**加分点**：另一个体验细节——点停止后不能等打字机把积压的字慢慢吐完，
要立即调收尾函数，否则「按了停止还在打字」。

**代码**：[`frontend/src/App.tsx`](../frontend/src/App.tsx) 的 `finishTyping` / `stopGenerating`

---

### Q5 中断保存「已生成的部分内容」，这个写操作要注意什么？

**考察**：幂等意识。

**参考答案**

必须**幂等**。中断保存可能被重复触发：用户连点停止、网络抖动导致重连、
前端重试逻辑。用 `INSERT` 会主键冲突报错或插出重复行。

```sql
CREATE TABLE chat_turns (
  session_id UUID, turn_index INT, content TEXT, status TEXT,
  PRIMARY KEY (session_id, turn_index)     -- 主键即幂等的基础
);
```

```sql
INSERT INTO chat_turns (...) VALUES (...)
ON CONFLICT (session_id, turn_index) DO UPDATE SET content = EXCLUDED.content, ...
```

**加分点**：指出主键选 `(session_id, turn_index)` 而不是自增 ID——业务上"同一会话
的第 N 轮"本来就唯一，用它做主键天然去重，不需要额外的去重逻辑。

**代码**：[`src/postgres.py`](../src/postgres.py) 的 `save_turn`

---

### Q6 ⭐ 「上下文感知恢复」能做到从上次中断的 token 继续生成吗？

**考察**：是否清楚模型 API 的实际能力边界，还是在背概念。

**参考答案**

**做不到字面意义的续传。** Ollama、Claude CLI、智谱 GLM 都**没有**「从半截继续生成」
的 API——这不是实现难度问题，是接口不存在。

真实的"恢复"是：把已生成的部分文本**当作上下文**，发起一次**新的**生成请求让模型接着写。
所以准确说法是 **re-enter（重新进入循环）而不是 resume（续传）**。

**加分点**：区分两种"恢复"的价值。
本项目的持久化恢复的是**给人看的历史记录**（刷新页面对话还在），
不是给模型的上下文——因为这个项目的 prompt 是单轮无状态的，压根不喂历史给模型。
所以"中断导致模型上下文变脏"这个风险在本项目**不存在**。

---

### Q7 建对话历史表时有什么容易忽略的坑？

**考察**：全局视野，能否想到与已有代码的相互影响。

**参考答案**

M2 以前全量索引会 `DROP TABLE`；如果把对话历史表和派生索引放在一起管理，用户点
一次「重新整理书架」，**聊天记录全没了**。M2 已改为单书事务替换，不再走全库
DROP，但边界仍然成立：对话是用户数据，索引是可以重算的派生数据，生命周期不同。

所以对话表独立用 `CREATE TABLE IF NOT EXISTS` 建，不参与索引 schema 和替换事务。

**加分点**：持久化失败不该让核心功能不可用——本项目里建表失败只打日志、不阻断启动；
保存失败只打日志、不影响本次回答返回。会话持久化是增强功能，不是关键路径。

---

## 二、前端渲染性能

### Q8 ⭐⭐ LLM 流式输出时，UI 怎么保证不卡？

**考察**：这是最常见的流式 UI 题，但多数人只答"节流"。

**参考答案**（一句话：**少重绘、轻节点、按需交互**，不是"更快地重绘"）

本项目的真实翻车过程：一开始按 **60fps**（16ms 一刷）追求丝滑，结果长回答明显卡顿。
三个原因叠加：

| 问题 | 为什么致命 |
| --- | --- |
| 每 16ms `setState` | React 重渲染**整个对话所有气泡**，不只是正在打字那条 |
| 每条气泡有 5 个 antd 省略号组件 | 这类组件每次渲染要**同步测量 DOM**，5×60次/秒×N条历史 = 布局狂抖 |
| 自动滚动绑在消息变化上 | 60 次/秒 `scrollTo`，每次强制重排 |

**对话越长越卡**——历史气泡数量在增长，而它们本不该重渲染。

三个修法：

```ts
// ① 降帧率 + 追赶：打字机不需要 60fps
const TICK_MS = 33;             // 30fps，重渲染次数砍半
const MIN_CHARS_PER_TICK = 2;
const CATCH_UP_TICKS = 42;      // 积压越多吐越快，防止字幕越落越远

// ② memo 隔离
const Sources = memo(...)              // 5 个测量组件，流式期间零重渲染
export default memo(MessageBubble)     // 历史气泡不跟着新回答重渲染

// ③ 滚动：只在贴底时跟随 + rAF 合帧
const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
if (!nearBottom) return;               // 用户在翻历史，别打断他
requestAnimationFrame(() => { el.scrollTop = el.scrollHeight });
```

**`memo` 生效的前提**：更新函数只替换数组最后一项，其余对象引用不变。
如果每次都 `messages.map(...)` 造新对象，`memo` 全部失效——这点很多人写错。

**加分点**：主动说出「顺带解决了后端出字粒度不一致」——Ollama 逐 token 返回、
GLM 常常一两个大 chunk 就是全文，前端统一排队按字吐出，观感一致。
这正是「UI 不必和 token 同频」的实践。

**代码**：[`App.tsx`](../frontend/src/App.tsx)、[`MessageBubble.tsx`](../frontend/src/components/MessageBubble.tsx)

---

### Q9 ⭐ 怎么证明你的渲染优化真的有效？

**考察**：有没有量化意识。说"感觉流畅了"是不合格的。

**参考答案**

用 `PerformanceObserver` 监听 **longtask**（>50ms 的主线程阻塞任务，是卡顿的直接来源）：

```js
window.__lt = [];
new PerformanceObserver(l => l.getEntries().forEach(e => __lt.push(e.duration)))
  .observe({ entryTypes: ["longtask"] });
// 跑完一次完整流式回答后
JSON.stringify({ 次数: __lt.length, 超100ms: __lt.filter(d => d > 100).length });
```

本项目实测：**GLM-4.6 连续输出 90 秒以上，长任务数 = 0**。

**加分点**：区分「前端卡顿」和「后端出字慢」。本项目测试时发现 90 秒只出了 250 字，
一开始怀疑还是卡——查下来是 GLM-4.6 是推理模型，先花很长时间在 reasoning 阶段，
正文是断续吐出来的。**longtask 为 0 说明前端没问题，慢是模型的特性。**
不做这个区分会把优化方向搞错。

**清单**：[`tests/PERF_CHECK.md`](../tests/PERF_CHECK.md)（6 条针对不同渲染压力的测试问题）

---

## 三、RAG 检索质量

### Q10 ⭐⭐ 用户说"检索不准"，你怎么定位？

**考察**：系统性排查能力。这题在本项目里有个非常典型的案例。

**参考答案**

本项目的真实案例：用户问「韩立的结局」，系统答「根据提供的片段无法确定」。
排查下来是**三个独立原因叠加**，任何一个不修都不行：

**① 切分失效（真 bug，影响全局）**

小说 txt 用**单换行**分段，但切分代码按**空行**切。《凡人修仙传》有 14 万个 `\n`，
只有 2468 个空行 → 切出 2468 个**平均 3108 字**的巨块，是设定值 500 的 6 倍。

致命之处：`bge-small-zh` 的 max_seq_length 是 **512 token**。
**每个片段只有开头约 500 字进了向量，后面 84% 的内容在检索中完全不可见**——
文本在库里，但永远命不中。

修完：2468 → **19501 个片段**，平均 463 字，零超限。

**② 「结局」是结构性问题，语义检索答不了**

结局那段原文里**不会出现"结局"二字**，向量相似度自然匹配不上。
需要额外的「结构性召回」：直接按 `chunk_id` 取书的首/尾片段，参与融合排序。

**③ 文件末尾不是正文结局**

文件名写着「全本**+番外**」——正文大结局在 `#19472`「(全书完)」，
**之后 28 个片段全是番外和作者后记**。纯按"文件最末"取，拿到的是番外内容。
需要优先定位「大结局/全书完/尾声」标记。

**加分点**：说出排查顺序——先确认**数据在不在库里**（我一开始误判成"只索引了 16%"，
查了总字数才发现文本都在、是切分尺寸的问题），再看**检索能不能命中**，
最后才看**模型会不会用**。顺序错了会修错地方。

**代码**：[`src/loader.py`](../src/loader.py)、[`src/rag.py`](../src/rag.py)

---

### Q11 ⭐ embedding 模型有长度上限，你的切分策略怎么保证不超？

**考察**：是否知道"切分尺寸"和"模型上限"必须挂钩，而不是随便设个数。

**参考答案**

光设 `CHUNK_SIZE` 不够，要**硬保证**每个片段都不超过上限，否则超出部分静默丢失
（不报错，只是检索不到，极难发现）。

两层保护：

1. 按**任意换行**切分，不只按空行——兼容不同来源的文本格式
2. 单个段落就超长时，按**句末标点**二次切分（`。！？…；`），保住语义；
   实在没标点可断（超长无标点文本）就硬切

```python
for piece in _split_long_paragraph(para):   # 先拆超长段落
    if len(candidate) <= CHUNK_SIZE:        # 再聚合，硬保证不超
        ...
```

**加分点**：中文场景下"字数"和"token 数"不是 1:1，`CHUNK_SIZE=500` 字对
512 token 的模型是偏保守但安全的选择；如果要压榨空间需要真的用 tokenizer 计数。

---

### Q12 用户把书名打错字（"诡秘之主"→"闺蜜之主"）也要能查到，怎么做？

**考察**：容错设计，以及"别过度容错"的判断。

**参考答案**

精确匹配失败时，用**编辑距离**（Levenshtein）做兜底：在问题里滑动一个与书名等长的
窗口，某个窗口与书名距离在容差内就算命中。

容差按标题长度取，不能一刀切：

- 3 字标题最多错 1 个字
- 4 字及以上最多错 2 个字（"闺蜜之主"→"诡秘之主"正好错 2 个）

**关键是要测负例**：不同的书之间不能互相误判。我写了 17 条用例，
一半是"不该匹配上"的（`凡人修仙传的结局` 不能匹配到 `《诡秘之主》`）。

**加分点**：容错只用于**书名这种短的、封闭集合的**字段。不要对全文检索做模糊匹配
——那会让召回噪音爆炸。

**代码**：[`src/rag.py`](../src/rag.py) 的 `_fuzzy_contains`

---

### Q13 只提人物名（"韩立的结局"）不提书名，怎么知道用户问的是哪本书？

**考察**：多路信号融合。

**参考答案**

分优先级：

1. 问题里直接写了书名 → 只查那本
2. 否则用**已有的召回结果**反推：哪本书命中的片段最多，就是它
3. 都没有 → 每本书都取一点

第 2 条是关键——不需要额外的分类模型，语义/关键词召回的结果本身就是信号。

```python
hint_novels = named_novels or _dominant_novels(semantic + keyword)
positional = self.positional_retrieve(question, hint_novels=hint_novels)
```

**加分点**：留跨书的余地——如果第二名的命中数达到第一名的一半以上就一并保留，
因为用户可能真的在问"两本书的结局有什么不同"。

---

## 四、工程实践与 CI

### Q14 ⭐⭐ CI 里用 bot 自动创建 PR，为什么 PR 上一个检查都没有？

**考察**：GitHub Actions 的安全模型。这题很多人（包括我）会先猜错。

**参考答案**

GitHub 的**防递归设计**：用默认 `GITHUB_TOKEN` 创建的 PR/push
**不会触发任何新的 workflow run**。目的是防止 workflow 自己触发自己无限循环。

后果：如果 main 分支保护要求"必需检查通过才能合并"，bot 开的 PR 会**永久卡死**
——检查永远不会出现，`mergeStateStatus: BLOCKED`。

官方给的解法（都有代价）：

| 方案 | 代价 |
| --- | --- |
| 专用 PAT 或 GitHub App token | 多一个长期凭证要管理和轮换 |
| 手动 Close 再 Reopen PR | 还是手工操作 |
| 关掉 `enforce_admins` 强制合并 | 削弱分支保护 |

**加分点（也是我踩的坑）**：我一开始以为"让 CI 也监听 bot 分支的 `push` 事件"
能绕过去——**错的**，`push` 和 `pull_request` 同样受这条限制。
这个误判让我白改了一个 PR。查官方文档才确认。

**最终决策**：这个项目手动跑一条命令 1 秒钟，**不值得**为此引入长期凭证或削弱保护，
所以删掉了自动化。**"不做"也是一种设计决策**，前提是说清代价对比。

---

### Q15 分支保护设了「必需检查通过」，为什么你自己也 push 不上去了？

**考察**：是否真的动手配过分支保护。

**参考答案**

「必需检查通过」的判定依据是"**这个 commit 有没有已通过的 CI 记录**"。
直接 `git push` 一个新 commit 时，它从没在 GitHub 上跑过 CI，天然没有通过记录
→ 被无条件拒绝（`GH006`），不管代码好不好。

```
remote: - 2 of 2 required status checks are expected.
 ! [remote rejected] main -> main (protected branch hook declined)
```

所以「必需过 CI」和「能直推 main」是**互斥**的，鱼和熊掌不可兼得——
这不是配置没调好，是机制本身如此。要 CI 把关就必须走 PR 流程。

**加分点**：`enforce_admins: true` 会让规则对仓库管理员也生效。
不加这个，保护对自己形同虚设。

---

### Q16 CHANGELOG 自动生成，用 `--prepend` 有什么问题？

**考察**：细节。看似能用的方案在长期使用中会出问题。

**参考答案**

`--prepend` 是"往文件最前面硬插一整块"，它**不认识哪里是手写内容**。后果：

- 每次插入都带上工具自己生成的标题 → `# 更新日志` 重复出现
- 手写的格式说明和历史记录被越挤越下面，跑几次结构就乱了

正确做法：生成到**临时文件**，再用脚本**只替换「未发布」那一段**（定位
`## [未发布]` 到下一个 `## [` 之间），手写内容一字不动。

```bash
git cliff --unreleased --strip header --output /tmp/unreleased.md
python3 scripts/merge_changelog.py /tmp/unreleased.md CHANGELOG.md
```

**加分点**：这个合并脚本要**幂等**——连跑两次结果必须一致（我实测过 md5 相同），
否则 CI 里重跑会累积垃圾。

**代码**：[`scripts/merge_changelog.py`](../scripts/merge_changelog.py)

---

### Q17 依赖锁版本时，`^1.50.0` 和 `1.50.0` 有什么区别？什么时候必须锁死？

**考察**：依赖管理的实际经验。

**参考答案**

`^1.50.0` 允许升级到任何 `1.x`。本项目踩过：Playwright 最新版要求 **Node 20+**，
而项目文档承诺 Node 18+，机器上装的也是 18。如果写 `^`，下次 `npm install`
可能悄悄升到不兼容版本，然后 CI 或本地莫名报错。

**必须锁死的场景**：依赖的**运行环境要求**（Node/Python 版本）比你的项目更严格时。
这种不兼容不体现在 API 上，靠类型检查发现不了。

```json
"@playwright/test": "1.50.0"   // 不带 ^，配注释说明为什么
```

**加分点**：锁死时要在代码或文档里**写清原因**，否则后人会以为是遗漏，随手升上去。

---

## 五、测试设计

### Q18 ⭐ 给一个依赖大模型和数据库的应用写 e2e 测试，怎么设计？

**考察**：测试边界的划分能力。

**参考答案**

**把外部依赖全部 mock 掉，只测前端渲染逻辑。** 本项目的 e2e 用 `page.route`
拦截所有 `/api/*` 请求，返回固定假数据：

- **不需要** PostgreSQL、Ollama、任何云端 key
- **不消耗** 真实账号额度
- 结果**确定**，不会因为模型这次答得不一样就失败
- CI 里跑得快（12 个用例 6 秒）

```ts
await page.route("**/api/ask", route => route.fulfill({
  contentType: "text/event-stream",
  body: buildSseBody({ tokens: ["雾隐", "山庄", "的庄主是", "顾长风"] }),
}));
```

**加分点**：说清"这样测的是什么、不是什么"。
这套测试**不验证** RAG 检索质量和模型回答正确性——那需要另一套评测
（本项目用 [`tests/qa_test_set.json`](../tests/qa_test_set.json) 单独做）。
把两件事混在一起会得到一套又慢又不稳定的测试。

**代码**：[`frontend/e2e/mock-api.ts`](../frontend/e2e/mock-api.ts)

---

### Q19 ⭐ 怎么测「点停止按钮」这种时序相关的交互？

**考察**：异步测试设计。

**参考答案**

难点：普通 mock 一次性返回完整响应，请求瞬间结束、界面立刻收尾，
**根本来不及点停止**。

解法：让 mock handler **不调用 `fulfill()`**，请求就一直 pending，
界面稳定停在"生成中"，可以从容点停止按钮。前端 `abort()` 后请求被取消，
handler 里的等待随之作废——这正是真实的中断路径。

```ts
await page.route("**/api/ask", async () => {
  // 故意不 fulfill：请求保持 pending，直到被 abort
});
```

**加分点**：不要用 `waitForTimeout(固定毫秒)` 来等状态——用
`expect(locator).toBeVisible()` 这类**自带轮询重试**的断言，
测试才不会因为机器快慢而随机失败。

**代码**：[`frontend/e2e/interrupt-and-resume.spec.ts`](../frontend/e2e/interrupt-and-resume.spec.ts)

---

### Q20 你的测试挂了，怎么判断是代码有 bug 还是测试写得不对？

**考察**：调试思路。本项目有个很典型的例子。

**参考答案**

本项目案例：断言"出处卡片有『展开』链接"失败，报 `element(s) not found`。

**不是代码 bug，是 mock 数据不真实**：我造的假原文太短，在桌面宽度下一行就放得下，
antd 的省略号组件判断"不需要截断"，压根没渲染「展开」链接。

判断方法：**看失败的断言在真实环境下成不成立**。手动打开页面，
真实数据下「展开」是正常出现的 → 说明产品代码没问题，是测试数据不够真实。

修法是让 mock 文本足够长，并**在代码注释里写明原因**，避免后人又改短：

```ts
// 注意：文本要足够长（超过一行），否则 antd 省略号组件不会渲染"展开"链接——
// 之前踩过这个坑。
```

**加分点**：区分三类测试失败——①产品 bug ②测试数据/断言不对 ③环境问题（依赖版本、
端口占用）。三类的修法完全不同，先分类再动手。

---

## 附：这些题的来源

全部出自 [`novel-rag`](https://github.com/ningkaikok/novel-rag) 这个项目的真实开发过程。
相关文档：

- [`docs/streaming-interview-notes.md`](streaming-interview-notes.md) — 流式中断与 UI 性能的详细展开
- [`tests/PERF_CHECK.md`](../tests/PERF_CHECK.md) — 渲染性能自查清单
- [`CHANGELOG.md`](../CHANGELOG.md) — 每个改动的用户视角描述
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — 开发、测试、提交和发布规范
- [`AGENTS.md`](../AGENTS.md) — AI 编程 Agent 的统一项目入口

**面试建议**：这些题的价值不在"标准答案"，而在**能说出验证过程和踩过的坑**。
比如 Q4（abort 后 reader 不一定抛异常）、Q14（我误判了 push 事件能绕过限制），
这类"我原本以为 X，实测发现 Y"的经历，比背对答案更能证明真的做过。
