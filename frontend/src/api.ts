// 与 FastAPI 后端通信的封装。开发期 /api 由 Vite 代理到 http://localhost:8000

export interface Source {
  novel: string;
  chunk_id: number;
  text: string;
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
  if (!res.ok) throw new Error((await res.json()).detail ?? "上传失败");
}

export async function deleteBook(name: string): Promise<void> {
  const res = await fetch(`/api/books/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error((await res.json()).detail ?? "删除失败");
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
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "获取模型列表失败");
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
  onSources?: (sources: Source[]) => void;
  onToken?: (token: string) => void;
  onDone?: () => void;
  onError?: (err: Error) => void;
}

/** 发起提问并消费 SSE 流：先收到 sources，再逐 token 收 answer。 */
export async function askStream(
  question: string,
  topK: number,
  handlers: AskHandlers
): Promise<void> {
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
    });
    if (!res.ok || !res.body) {
      throw new Error((await res.json().catch(() => ({}))).detail ?? "请求失败");
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

function handleEvent(raw: string, handlers: AskHandlers) {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  if (event === "sources") handlers.onSources?.(JSON.parse(data));
  else if (event === "token") handlers.onToken?.(JSON.parse(data));
}
