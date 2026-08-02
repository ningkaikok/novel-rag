# 流式输出的两个面试题 · 结合本项目的解释

两道题都围绕「LLM 流式输出」，但方向相反：

- **第一题问"怎么停下来、怎么接着来"** —— 中断与恢复（本项目**尚未实现**，下面是设计）
- **第二题问"怎么不卡"** —— UI 渲染性能（本项目**已实现并实测过**，有真实数据）

---

## 第二题：流式输出时怎么保证 UI 性能

**先讲这题，因为本项目真做过，而且踩过坑。**

### 一句话

不是"更快地重绘"，而是**少重绘、轻节点、按需交互**。

### 本项目的真实翻车过程

刚加打字机效果时，我图"丝滑"设成了 **60fps**（每 16ms 刷一次）。结果长回答明显卡顿。
根因是三件事叠加：

| 问题 | 为什么致命 |
| --- | --- |
| 每 16ms 就 `setMessages` | React 会重渲染**整个对话的所有气泡**，不只是正在打字那一条 |
| 每条气泡里有 5 个 antd `Typography.Paragraph` 省略号组件 | 这类组件每次渲染都要**同步测量 DOM**（算"展开"按钮位置），5 个 × 60 次/秒 × N 条历史 = 布局狂抖 |
| 自动滚动绑在 `messages` 变化上 | 60 次/秒 `scrollTo`，每次强制重排 |

**对话越长越卡** —— 因为历史气泡数量在增长，而它们本不该重渲染。

### 三个修法（都在代码里）

**① 降帧率 + 追赶机制** — [`App.tsx`](../frontend/src/App.tsx)

```ts
const TICK_MS = 33;            // 60fps → 30fps，重渲染次数直接砍半
const MIN_CHARS_PER_TICK = 2;  // 每次至少吐 2 个字
const CATCH_UP_TICKS = 42;     // 积压越多吐越快，约 1.4s 内清空
```

打字机**不需要 60fps**。30fps 观感一样是匀速打字，成本减半。
`CATCH_UP_TICKS` 是防止"模型吐得比 UI 快，字幕越落越远"。

> 顺带解决了另一个问题：后端出字粒度不统一（Ollama 逐 token、GLM 常常一两个大 chunk
> 就是全文）。前端统一排队按字吐出，观感一致——这就是"UI 不必跟 token 同频"的实践。

**② `React.memo` 隔离不该重渲染的部分** — [`MessageBubble.tsx`](../frontend/src/components/MessageBubble.tsx)

```ts
const Sources = memo(...)       // 出处卡片：5 个测量组件，流式期间完全不重渲染
const Thinking = memo(...)      // 思考过程面板：trace 在一次回答里不变
export default memo(MessageBubble)  // 整条气泡：历史消息不跟着新回答重渲染
```

关键前提：`patchLast` 只替换数组最后一项，其余 `msg` 引用不变，`memo` 才生效。

**③ 滚动节流 + 只在贴底时跟随**

```ts
const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
if (!nearBottom) return;                      // 用户在翻历史，别打断他
const id = requestAnimationFrame(() => { ... }); // 合并同一帧内的多次触发
```

### 实测结果

用 `PerformanceObserver` 监听 `longtask`（>50ms 的主线程阻塞任务）：

```js
new PerformanceObserver(l => l.getEntries().forEach(e => __lt.push(e.duration)))
  .observe({ entryTypes: ["longtask"] });
```

**GLM-4.6 连续输出 90 秒以上，长任务数 = 0。** 完整清单见
[`tests/PERF_CHECK.md`](../tests/PERF_CHECK.md)（6 条针对不同渲染压力的测试问题）。

### 面试怎么答

> "我踩过这个坑：一开始按 60fps 刷，长回答明显卡。定位到是每帧 `setState` 导致
> 整个列表重渲染，而列表里有需要同步测量 DOM 的省略号组件。三个修法：帧率降到
> 30fps（打字机不需要 60fps）、用 `memo` 让历史消息和重型子组件不参与重渲染、
> 滚动改成 rAF 合帧且只在贴底时跟随。实测持续输出 90s 长任务为 0。"

**加分点**：主动说出"顺带解决了后端出字粒度不一致的问题"——前端排队按字吐，
把 Ollama 的逐 token 和 GLM 的整段返回统一成一致观感。

---

## 第一题：中断与上下文感知恢复

> ✅ **已实现**（2026-08-02）。下面每条都有对应代码和实测结果。

### 那句话拆开看

> "生成是带检查点的后台状态机，传输是可断线重连的事件流，
> 取消是全链路协作式信号，恢复是 partial 上下文 + 用户新意图重新进 loop。"

四个词对应四层，各管一件事：

| 层 | 关键词 | 管什么 | 本项目现状 |
| --- | --- | --- | --- |
| 生成 | 检查点、状态机 | 生成到哪一步了，能不能存档 | ✅ `chat_turns` 表存 partial + `interrupted` 状态 |
| 传输 | 事件流、断线重连 | 数据怎么送到浏览器，断了怎么办 | ⚠️ SSE + 中断可用；**断线自动重连未做** |
| 取消 | 协作式信号 | 点 Stop 之后，整条链路怎么一起停 | ✅ AbortSignal → 断连检测 → 停止索取 token |
| 恢复 | partial + 新意图 | 下次接着来时，上下文怎么拼 | ✅ 刷新按 sessionId 拉回历史（含 partial） |

### 为什么是"协作式"而不是"强杀"

这是最容易答错的点。**"取消"不是把线程 kill 掉**，而是一路传递一个"该停了"的信号，
每一层自己找安全点收手：

```
用户点 Stop
  ↓ AbortController.abort()          浏览器层：停止读 SSE 流
  ↓ 连接断开
  ↓ request.is_disconnected()        后端层：检测到客户端走了
  ↓ break 出生成循环 + 关闭上游连接   停止向 Ollama/GLM 要下一个 token
  ↓ 写入检查点（已生成的部分 + 状态） 存档层：留下"到这为止"的痕迹
```

**为什么强杀不行**：强杀会留下脏状态——数据库写一半、上游连接没关（还在扣你的
GLM token 额度）、消息状态卡在 `streaming: true` 永远收不到结束信号。

### 本项目现在的真实缺口

**① 前端：AbortController** — [`api.ts`](../frontend/src/api.ts) 的 `askStream` 接受
`signal`，[`App.tsx`](../frontend/src/App.tsx) 每次提问建一个 `AbortController` 存进 ref，
生成中发送键变成「■ 停止」，点它调 `abort()`。

**② 后端：async + 断连检测** — [`main.py`](../backend/main.py) 的 `ask` 改成了
`async def`，`event_stream` 循环里每吐一个 token 就 `await request.is_disconnected()`，
断了就 `token_iter.close()` 并跳出。

三个生成后端（Ollama / Claude CLI / GLM）都是**同步**生成器，直接在协程里 `for` 会阻塞
事件循环、导致断连检查永远轮不到执行。所以用 `run_in_threadpool` 逐个取 token，
每个 `await` 都是一次让出控制权的机会：

```python
chunk = await run_in_threadpool(_next_or_sentinel, token_iter)
if chunk is _SENTINEL:
    break
...
if await request.is_disconnected():
    interrupted = True
    break
```

> 用哨兵而不是捕获 `StopIteration`：后者不能穿过 `await` 边界（会变成 `RuntimeError`）。

**③ 检查点：`chat_turns` 表** — [`postgres.py`](../src/postgres.py) 的
`ensure_chat_schema()`。刻意和 `recreate_schema()` 分开——那个函数重建向量索引时会
`DROP TABLE`，聊天记录不该因为「重新整理书架」被清空。

**④ 幂等：主键 + UPSERT** — `(session_id, turn_index)` 做主键，`save_turn()` 用
`ON CONFLICT DO UPDATE`。中断保存可能被重复触发（连点停止、网络抖动），
覆盖同一行而不是插重复数据。

### 实测结果

**中断真的省下了额度**（这是整件事的核心价值）。用 `curl --max-time 4` 提前断开，
然后隔 12 秒观察数据库：

```
断开瞬间：  还没写库
12 秒后：   102 字 / interrupted   ← 不再增长
```

如果上游没停，GLM 会继续吐几千字。**内容停在 102 字不动，证明协作式取消生效了。**

浏览器端验证（浅色/深色都测过）：点停止后出现「已停止生成」标记、光标消失、
停止键换回发送键、**没有 ⚠️ 报错图标**（主动停止不是错误）、已生成内容保留；
刷新页面后提问/回答/出处/思考过程全部恢复，被中断那轮仍标记为已停止。

e2e 覆盖 5 条用例（[`interrupt-and-resume.spec.ts`](../frontend/e2e/interrupt-and-resume.spec.ts)）：
停止按钮出现与生效、主动停止 vs 真出错的区分、历史恢复、中断状态恢复、无历史时显示欢迎页。

### 踩到的一个真坑

第一版只在 `onError` 里处理中断标记，结果点了停止**界面卡在生成中**——光标还转、
停止键还在。原因是 **`abort()` 后 `reader.read()` 不一定抛异常，可能直接返回
`done: true`**，于是走的是正常结束路径（`onDone` → `finishTyping`）而不是 `onError`。

修法：`finishTyping()` 里也读 `userStoppedRef`。另外让 `stopGenerating()` 直接调
`finishTyping()` 立即收尾——不然打字机会把积压的字慢慢吐完，用户按了停止却还在打字。

### 关于 tool_call 半截的问题

面试官提到的「`Pending tool calls exist without results`」在**本项目不会发生**——
这个项目是纯 `检索 → 生成`，没有 agentic tool loop（`grep tool_use` 结果为空）。

但**同构的问题存在**：SSE 事件序列 `trace → sources → token* → done` 本身是隐式协议。
中断发生在 `token` 和 `done` 之间时，消息对象会卡在 `streaming: true` 永远等不到 `done`
——这就是"半截消息链"的变体，主语从 tool_call 换成了 SSE 事件。

好消息：**这条路径基本已经兜住了**。`askStream` 的 `try/catch` 会把 `AbortError` 交给
`onError`，里面已经把 `streaming` 置 `false`、保留已有内容。唯一要改的是**文案**——
现在统一显示 `⚠️ ${e.message}`，用户主动点 Stop 不该显示成报错，得判断
`e.name === 'AbortError'` 显示"已停止"。

### 一个必须说清的现实限制

**"断点续传"在这个项目做不到字面意义上的实现。** Ollama / Claude CLI / 智谱 GLM
三个后端**都没有"接着上次生成到一半的 token 继续吐"的 API**。

真实的"恢复"是：把已生成的部分文本**当作上下文的一部分**，发起一次**新的**生成请求，
让模型接着往下写。这就是那句话里「**partial 上下文 + 用户新意图重新进 loop**」的含义
——不是 resume，是 **re-enter**。

> 补充一个本项目的特殊情况：[`rag.py`](../src/rag.py) 的 `build_prompt` 目前是
> **完全无状态、单轮的**——每次生成只用当前问题 + 检索到的原文，不喂历史对话。
> 所以"中断导致模型上下文变脏"这个风险目前**不存在**。
> 持久化的意义是**给人看的历史记录**，不是给模型的上下文。

---

## 两题的共同点

都在回答同一个问题：**流式输出是一个"过程"，不是一次"结果"，所以每一层都要能处理
"中间状态"。**

- UI 层：中间状态会来几百次 → 别每次都全量重绘（✅ 30fps 节流 + memo 隔离）
- 传输层：中间状态可能断在任何一刻 → 要能优雅收场（✅ AbortSignal + 断连检测）
- 生成层：中间状态要能存档 → 检查点（✅ `chat_turns` 存 partial + status）
- 数据层：中间状态可能被重复写入 → 幂等（✅ 主键 + `ON CONFLICT DO UPDATE`）

### 还没做的

**断线自动重连**。现在网络断了（不是用户主动停）不会自动续上，只会当作一次中断
收尾。要做需要给 SSE 事件加序号，重连时带 `Last-Event-ID` 让后端从那之后继续推——
但受限于「模型不支持从半截继续生成」，重连能恢复的只是**传输**，不是**生成**。
对本地单人使用的场景收益有限，所以没做。
