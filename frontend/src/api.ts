// 与 FastAPI 后端通信的封装。开发期 /api 由 Vite 代理到 http://localhost:8000。
//
// 这里有两类通信方式，代表 AI 应用常见的「控制面」和「数据面」：
// 1. 普通 JSON 请求：上传、切换模型、查询任务状态——一次请求对应一次完整响应。
// 2. SSE 流：问答过程中持续接收检索步骤、出处和 token——一个响应包含很多事件。
// 把协议细节集中在本文件后，React 组件只处理业务状态，不需要理解 HTTP 分帧。
//
// ── 类型契约流水线 ─────────────────────────────────────────────────────────
// REST 接口的请求/响应类型不再手写，统一来自生成文件 src/api-generated.ts：
//
//   backend/schemas.py（Pydantic 模型，类型的唯一事实来源）
//     → uv run python scripts/export_openapi.py   （后端模型变更后先跑这个）
//     → openapi.json
//     → cd frontend && npm run gen:api            （再生成 TS 类型）
//     → src/api-generated.ts（auto-generated，勿手改）
//
// 本文件用别名把生成类型导出成业务代码熟悉的旧名字，组件层零改动；
// 生成物里 Pydantic 可选字段是 `field?: T | null`（比手写版多了 null），
// 使用处以生成为准：读值时按可空处理，不要回头改生成文件。

import type { components } from './api-generated';

type Schemas = components['schemas'];

// 后端统一的错误响应形状：{"error": {"code": ..., "message": ...}}。
// 不管是业务代码主动抛的错误、还是 FastAPI 自己的请求校验错误、
// 还是完全没预料到的异常，都是这个形状，这里只用得到 message。
//
// 关于 ErrorCode：后端 errors.py 里确实有 StrEnum，但这个信封是异常处理器
// 手工拼的 JSONResponse，不是任何 Pydantic response_model，因此不出现在
// OpenAPI / api-generated.ts 里；前端目前也不消费 code，所以没有对应类型。
async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return body?.error?.message ?? fallback;
  } catch {
    return fallback;
  }
}

/** 出处：一段被引用的原文（生成物里的名字是 SourceItem）。 */
export type Source = Schemas['SourceItem'];

// 「思考过程」的一步：检索流水线里某个阶段的真实动作。
// 生成版比旧手写版多了评测用的扩展字段（variants/reasons/evidence_tokens 等），
// 以及 ms?: number | null——读耗时的地方本来就做了 ?? 0 / != null 兜底，无需改动。
export type TraceStep = Schemas['TraceStep'];

export type RetrievalCandidate = Schemas['RetrievalCandidate'];

export type AgentStep = Schemas['AgentStep'];

// 命名错位说明：生成物把「单条命中」叫 SearchMatch、「搜索响应信封」叫 SearchResult；
// 旧手写代码正好相反。别名按旧名字对号入座，调用方零感知。
export type SearchResult = Schemas['SearchMatch'];

export type SearchResponse = Schemas['SearchResult'];

// 索引任务的状态机取值。后端契约里 IndexTaskStatus.status 只是普通 string
// （schemas.py 写的是 `status: str`），OpenAPI 里没有字面量联合可导入，
// 所以这个联合保留手写：它描述的是前端真正会遇到的六种状态，
// 供 UI 判断与文档使用；等后端把它改成 enum/Literal 后再切换到生成物。
export type IndexTaskState =
  'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled';

export type IndexResult = Schemas['IndexResult'];

export type IndexTask = Schemas['IndexTaskStatus'];

/** 问答路径：自动判断、强制依据书架原文、或跳过检索直接自由回答。
 * 生成物里有同名字面量联合（Schemas['AnswerMode']），直接采用。
 */
export type AnswerMode = Schemas['AnswerMode'];

export async function searchBooks(
  query: string,
  book?: string,
  limit = 20,
  offset = 0,
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
    offset: String(offset),
  });
  if (book) params.set('book', book);
  const res = await fetch(`/api/search?${params.toString()}`);
  if (!res.ok) {
    throw new Error(await extractErrorMessage(res, '全文搜索失败'));
  }
  return await res.json();
}

export async function listBooks(): Promise<string[]> {
  const res = await fetch('/api/books');
  if (!res.ok) throw new Error('获取书架失败');
  return (await res.json()).books;
}

// 下面这组书架操作（上传/删除/同步）都只负责「发起」：后端把它们放进后台线程，
// 立即返回一个 IndexTask，真正的进度靠 App 里的轮询不断拉取。

export async function uploadBooks(files: FileList | File[]): Promise<IndexTask> {
  const form = new FormData();
  for (const f of Array.from(files)) form.append('files', f);
  const res = await fetch('/api/books', { method: 'POST', body: form });
  if (!res.ok) throw new Error(await extractErrorMessage(res, '上传失败'));
  return (await res.json()).task;
}

export async function deleteBook(name: string): Promise<IndexTask> {
  const res = await fetch(`/api/books/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await extractErrorMessage(res, '删除失败'));
  return (await res.json()).task;
}

export async function reindex(force = false): Promise<IndexTask> {
  const res = await fetch(`/api/reindex?force=${force ? 'true' : 'false'}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await extractErrorMessage(res, '索引同步失败'));
  return await res.json();
}

export async function getCurrentIndexTask(): Promise<IndexTask | null> {
  const res = await fetch('/api/index-tasks/current');
  if (!res.ok) throw new Error(await extractErrorMessage(res, '获取索引任务失败'));
  return await res.json();
}

export async function getIndexTask(taskId: string): Promise<IndexTask> {
  const res = await fetch(`/api/index-tasks/${encodeURIComponent(taskId)}`);
  if (!res.ok) throw new Error(await extractErrorMessage(res, '获取索引进度失败'));
  return await res.json();
}

export async function cancelIndexTask(taskId: string): Promise<IndexTask> {
  const res = await fetch(`/api/index-tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await extractErrorMessage(res, '取消索引任务失败'));
  return await res.json();
}

export async function retryIndexTask(taskId: string): Promise<IndexTask> {
  const res = await fetch(`/api/index-tasks/${encodeURIComponent(taskId)}/retry`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await extractErrorMessage(res, '重试索引任务失败'));
  return await res.json();
}

/** 模型列表接口的响应（生成物里的名字是 ModelList）。 */
export type ModelsInfo = Schemas['ModelList'];

export async function listModels(): Promise<ModelsInfo> {
  const res = await fetch('/api/models');
  if (!res.ok) throw new Error(await extractErrorMessage(res, '获取模型列表失败'));
  return await res.json();
}

export async function setModel(model: string): Promise<void> {
  const res = await fetch('/api/model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) throw new Error('切换模型失败');
}

// ── SSE 流式事件类型：刻意保留手写，不走生成流水线 ─────────────────────────
// OpenAPI 只描述「一问一答」的 REST JSON；/api/ask 与 /api/agent/ask 的响应是
// text/event-stream，一个连接里推很多个事件，事件协议（step → sources → token*）
// 是前后端隐式约定，没有 response_model 可导出，自然也不在 api-generated.ts 里。
// 所以这里的回调签名与事件负载类型保持手写，后端改动时需要同步检查本段。
export interface AskHandlers {
  /** 检索每完成一步就回调一次，用于把「思考过程」逐条点亮。 */
  onStep?: (step: TraceStep) => void;
  onTrace?: (trace: TraceStep[]) => void;
  onSources?: (sources: Source[]) => void;
  onToken?: (token: string) => void;
  onAgentStep?: (step: AgentStep) => void;
  onDone?: () => void;
  onError?: (err: Error) => void;
}

/** 发起提问并消费 SSE 流：检索期间逐条收 step，然后 sources，再逐 token 收 answer。
 *
 * signal 用于用户中断（Stop 按钮）：abort 后 fetch 抛 AbortError，
 * 同时连接断开，后端检测到就会停止向上游模型索取 token。
 * sessionId 传了就落库，刷新页面能恢复历史。
 */
export async function askStream(
  question: string,
  topK: number,
  handlers: AskHandlers,
  options: { signal?: AbortSignal; sessionId?: string; mode?: AnswerMode } = {},
): Promise<void> {
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: topK,
        session_id: options.sessionId,
        mode: options.mode ?? 'auto',
      }),
      signal: options.signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(await extractErrorMessage(res, '请求失败'));
    }

    // 网络层出错（包括用户 abort 触发的 AbortError）统一交给 onError，
    // 由上层区分「主动停止」和「真的失败」——本文件不做这个判断，
    // 因为只有 UI 层知道用户点没点过停止按钮。
    await consumeEventStream(res, handlers);
    handlers.onDone?.();
  } catch (err) {
    handlers.onError?.(err as Error);
  }
}

/** Agent Lab 使用独立端点，但沿用 sources/token/done，并多出 agent_step 事件。
 *
 * sessionId 传了就落库，刷新页面能恢复历史——之前这个端点没有这个参数，
 * Agent Lab 的每一次对话都是纯内存，刷新必然清空。
 */
export async function askAgentStream(
  question: string,
  handlers: AskHandlers,
  options: { signal?: AbortSignal; maxSteps?: number; sessionId?: string } = {},
): Promise<void> {
  try {
    const res = await fetch('/api/agent/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        max_steps: options.maxSteps ?? 5,
        session_id: options.sessionId,
      }),
      signal: options.signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(await extractErrorMessage(res, 'Agent Lab 请求失败'));
    }
    await consumeEventStream(res, handlers);
    handlers.onDone?.();
  } catch (err) {
    handlers.onError?.(err as Error);
  }
}

async function consumeEventStream(res: Response, handlers: AskHandlers) {
  const reader = res.body!.getReader();
  // 网络分片不等于 SSE 事件分片：一次 reader.read() 可能只拿到半个 JSON，
  // 也可能同时拿到多个事件。TextDecoder 的 stream:true 保留跨分片的 UTF-8
  // 字节状态，buffer 则保留尚未遇到空行终止符的半个 SSE 事件。
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    // 流正常结束（后端生成完毕关闭连接）走这里；用户 abort 则 read() 会抛
    // AbortError——两种情况最终都回到 askStream 的 onDone/onError 分支收尾。
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分隔事件。循环处理是因为一个网络包里可能粘着多个事件；
    // 最后不足一个事件的尾巴继续留在 buffer，等待下一次 read() 补齐。
    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      handleEvent(raw, handlers);
    }
  }
}

// 后端存下来的一轮对话（用于刷新页面后恢复）。
// 注意与旧手写版的差异：生成物里 role / status 是普通 string（后端就是 str），
// agent_steps 等字段变成可选；消费方（useChatStream）已按可空/宽类型处理。
export type StoredTurn = Schemas['StoredTurn'];

/** 读回某个会话的历史对话。会话不存在时返回空数组，不算错误。 */
export async function loadSession(sessionId: string): Promise<StoredTurn[]> {
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error('读取会话历史失败');
  return (await res.json()).turns ?? [];
}

// 解析单个 SSE 事件块（不含结尾空行）。事件协议是隐式约定的：
// step → (sources) → token* → 连接关闭，没有显式的 done 事件——
// 「生成结束」由连接关闭表达，所以 onDone 在 consumeEventStream 返回后才触发。
function handleEvent(raw: string, handlers: AskHandlers) {
  let event = 'message';
  let data = '';
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    // SSE 规范允许一个事件的 data 拆成多行 data: 逐行拼接；
    // 本后端虽然总是单行发送，这里仍按规范处理以保持健壮。
    else if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  // 没有负载的事件（如只用来保活的注释行）直接跳过
  if (!data) return;
  // step 是检索期间逐条推的（新），trace 是一次性整包（历史会话恢复走这条）
  if (event === 'step') handlers.onStep?.(JSON.parse(data));
  else if (event === 'agent_step') handlers.onAgentStep?.(JSON.parse(data));
  else if (event === 'trace') handlers.onTrace?.(JSON.parse(data));
  else if (event === 'sources') handlers.onSources?.(JSON.parse(data));
  else if (event === 'token') handlers.onToken?.(JSON.parse(data));
  // 未识别的事件类型静默忽略：后端将来加新事件时旧前端不会崩
}
