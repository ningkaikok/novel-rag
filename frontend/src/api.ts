// 与 FastAPI 后端通信的封装。开发期 /api 由 Vite 代理到 http://localhost:8000

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
  text: string;
}

// 「思考过程」的一步：检索流水线里某个阶段的真实动作
export interface TraceStep {
  step: string;
  detail: string;
  /** 本阶段耗时（毫秒）。历史会话里存的旧记录没有这个字段。 */
  ms?: number;
}

export interface SearchResult extends Source {
  match_count: number;
}

export interface SearchResponse {
  query: string;
  total: number;
  results: SearchResult[];
}

export async function searchBooks(
  query: string,
  book?: string,
  limit = 20,
  offset = 0
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
    offset: String(offset),
  });
  if (book) params.set("book", book);
  const res = await fetch(`/api/search?${params.toString()}`);
  if (!res.ok) {
    throw new Error(await extractErrorMessage(res, "全文搜索失败"));
  }
  return await res.json();
}

export async function listBooks(): Promise<string[]> {
  const res = await fetch("/api/books");
  if (!res.ok) throw new Error("获取书架失败");
  return (await res.json()).books;
}

export async function uploadBooks(files: FileList | File[]): Promise<void> {
  const form = new FormData();
  for (const f of Array.from(files)) form.append("files", f);
  const res = await fetch("/api/books", { method: "POST", body: form });
  if (!res.ok) throw new Error(await extractErrorMessage(res, "上传失败"));
}

export async function deleteBook(name: string): Promise<void> {
  const res = await fetch(`/api/books/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await extractErrorMessage(res, "删除失败"));
}

export async function reindex(): Promise<{ chunk_count: number }> {
  const res = await fetch("/api/reindex", { method: "POST" });
  if (!res.ok) throw new Error("重建索引失败");
  return await res.json();
}

export interface ModelsInfo {
  models: string[];
  current: string;
}

export async function listModels(): Promise<ModelsInfo> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error(await extractErrorMessage(res, "获取模型列表失败"));
  return await res.json();
}

export async function setModel(model: string): Promise<void> {
  const res = await fetch("/api/model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) throw new Error("切换模型失败");
}

interface AskHandlers {
  /** 检索每完成一步就回调一次，用于把「思考过程」逐条点亮。 */
  onStep?: (step: TraceStep) => void;
  onTrace?: (trace: TraceStep[]) => void;
  onSources?: (sources: Source[]) => void;
  onToken?: (token: string) => void;
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
  options: { signal?: AbortSignal; sessionId?: string } = {}
): Promise<void> {
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: topK,
        session_id: options.sessionId,
      }),
      signal: options.signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(await extractErrorMessage(res, "请求失败"));
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 以空行分隔事件
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        handleEvent(raw, handlers);
      }
    }
    handlers.onDone?.();
  } catch (err) {
    handlers.onError?.(err as Error);
  }
}

// 后端存下来的一轮对话（用于刷新页面后恢复）
export interface StoredTurn {
  turn_index: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  trace: TraceStep[] | null;
  status: "complete" | "interrupted";
}

/** 读回某个会话的历史对话。会话不存在时返回空数组，不算错误。 */
export async function loadSession(sessionId: string): Promise<StoredTurn[]> {
  const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error("读取会话历史失败");
  return (await res.json()).turns ?? [];
}

function handleEvent(raw: string, handlers: AskHandlers) {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  // step 是检索期间逐条推的（新），trace 是一次性整包（历史会话恢复走这条）
  if (event === "step") handlers.onStep?.(JSON.parse(data));
  else if (event === "trace") handlers.onTrace?.(JSON.parse(data));
  else if (event === "sources") handlers.onSources?.(JSON.parse(data));
  else if (event === "token") handlers.onToken?.(JSON.parse(data));
}
