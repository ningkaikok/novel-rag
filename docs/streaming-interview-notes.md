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

> ⚠️ **本项目目前没有实现这部分**，下面是"如果要做，在这个项目里长什么样"。
> 面试时如实说"设计过但没落地"比含糊带过更好。

### 那句话拆开看

> "生成是带检查点的后台状态机，传输是可断线重连的事件流，
> 取消是全链路协作式信号，恢复是 partial 上下文 + 用户新意图重新进 loop。"

四个词对应四层，各管一件事：

| 层 | 关键词 | 管什么 | 本项目现状 |
| --- | --- | --- | --- |
| 生成 | 检查点、状态机 | 生成到哪一步了，能不能存档 | ❌ 无状态，纯函数式一次性生成 |
| 传输 | 事件流、断线重连 | 数据怎么送到浏览器，断了怎么办 | ⚠️ 有 SSE，但断了不能重连 |
| 取消 | 协作式信号 | 点 Stop 之后，整条链路怎么一起停 | ❌ 没有 Stop 按钮 |
| 恢复 | partial + 新意图 | 下次接着来时，上下文怎么拼 | ❌ 刷新页面历史全丢 |

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

**① 前端没有 AbortSignal** — [`api.ts`](../frontend/src/api.ts) 的 `askStream` 用裸
`fetch()`，没传 `signal`。想中断只能关标签页。

**② 后端是同步生成器，客户端断了它还在跑** — [`main.py`](../backend/main.py) 的
`event_stream()` 是 `def` 不是 `async def`，FastAPI 丢进线程池。前端 `abort()` 之后，
这个任务**不会自动停**，会继续拿着 Ollama/GLM 的连接跑到底。

> 这不是锦上添花，是**防止"用户点了 Stop，你还在偷偷扣他 GLM 的费用"**。
> 要真正生效得改 `async def` + 循环里 `await request.is_disconnected()`。

**③ 没有 session/检查点** — 消息只存在 React 的 `useState` 里，刷新页面就没了。
数据库里只有 `novel_chunks`（小说片段），没有对话历史表。

**④ 幂等** — 目前 `/api/ask` 不写库（检索只读、生成不落盘），暂时没坑。但一旦加了
③ 的持久化，"存中断时的部分内容"这个写操作必须是 `UPSERT`，否则重试会主键冲突。

### 如果要做，长这样

```sql
-- 检查点表：一个 session 一串 turn
CREATE TABLE chat_turns (
  session_id  UUID,
  turn_index  INT,
  role        TEXT,           -- user / assistant
  content     TEXT,
  sources     JSONB,
  status      TEXT,           -- complete / interrupted
  PRIMARY KEY (session_id, turn_index)   -- 主键即幂等保障
);
```

```ts
// 前端：AbortController + Stop 按钮
const ctrl = new AbortController();
abortRef.current = ctrl;
await askStream(question, topK, handlers, ctrl.signal);
// Stop 按钮：abortRef.current?.abort()
```

```python
# 后端：改成 async，循环里检测断连
async def event_stream():
    async for chunk in token_iter:
        if await request.is_disconnected():
            save_checkpoint(session_id, partial_text, status="interrupted")
            break   # 同时关闭到 Ollama/GLM 的上游连接
        yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
```

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

- UI 层：中间状态会来几百次 → 别每次都全量重绘（**已做**）
- 传输层：中间状态可能断在任何一刻 → 要能重连、要能优雅收场（**部分**）
- 生成层：中间状态要能存档 → 检查点（**未做**）
- 数据层：中间状态可能被重复写入 → 幂等（**未做，但目前不写库所以没坑**）
