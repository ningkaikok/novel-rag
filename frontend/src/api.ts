// 与 FastAPI 后端通信的封装。开发期 /api 由 Vite 代理到 http://localhost:8000。
//
// 这里有两类通信方式，代表 AI 应用常见的“控制面”和“数据面”：
// 1. 普通 JSON 请求：上传、切换模型、查询任务状态——一次请求对应一次完整响应。
// 2. SSE 流：问答过程中持续接收检索步骤、出处和 token——一个响应包含很多事件。
// 把协议细节集中在本文件后，React 组件只处理业务状态，不需要理解 HTTP 分帧。

// 后端统一的错误响应形状：{"error": {"code": ..., "message": ...}}。
// 不管是业务代码主动抛的错误、还是 FastAPI 自己的请求校验错误、
// 还是完全没预料到的异常，都是这个形状，这里只用得到 message。
async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return body?.error?.message ?? fallback;
  } catch {
    return fallback;
  }
}

export interface Source {
  novel: string;
  chunk_id: number;
  /** 旧索引或没有规范章节标题的 txt 可能为空。 */
  chapter_title?: string | null;
  text: string;
}

// 「思考过程」的一步：检索流水线里某个阶段的真实动作
export interface TraceStep {
  step: string;
  detail: string;
  /** 本阶段耗时（毫秒）。历史会话里存的旧记录没有这个字段。 */
  ms?: number;
  /** 稳定的机器可读阶段名，用于评测面板，不依赖中文展示文案。 */
  stage_key?: string | null;
  candidates?: RetrievalCandidate[];
}

export interface RetrievalCandidate {
  novel: string;
  chunk_id: number;
  chapter_title?: string | null;
  rank: number;
  score?: number | null;
  score_label?: string | null;
  previous_rank?: number | null;
  selected?: boolean;
}

export interface AgentStep {
  step: number;
  reason: string;
  tool: string;
  args: Record<string, unknown>;
  observation: string;
  source_ids: string[];
}

export interface SearchResult extends Source {
  match_count: number;
}

export interface SearchResponse {
  query: string;
  total: number;
  results: SearchResult[];
}

export type IndexTaskState =
  'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled';

export interface IndexResult {
  novels: string[];
  chunk_count: number;
  added: string[];
  modified: string[];
  deleted: string[];
  unchanged: string[];
  contextualized: number;
  relations: number;
  hierarchy_nodes: number;
}

export interface IndexTask {
  id: string;
  status: IndexTaskState;
  stage: string;
  progress: number;
  message: string;
  error?: string | null;
  force: boolean;
  retry_of?: string | null;
  result?: IndexResult | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

/** 问答路径：自动判断、强制依据书架原文、或跳过检索直接自由回答。 */
export type AnswerMode = 'auto' | 'grounded' | 'free';

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

export interface ModelsInfo {
  models: string[];
  current: string;
}

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

// 后端存下来的一轮对话（用于刷新页面后恢复）
export interface StoredTurn {
  turn_index: number;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[] | null;
  trace: TraceStep[] | null;
  // 只有 Agent Lab 那条链路的对话会有这个字段；普通问答模式恒为 null。
  agent_steps: AgentStep[] | null;
  status: 'complete' | 'interrupted';
}

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
