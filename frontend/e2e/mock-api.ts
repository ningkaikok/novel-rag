import type { Page } from "@playwright/test";

/**
 * e2e 测试只验证前端渲染逻辑，所有 /api/* 请求都在这里拦截、返回固定假数据。
 * 不依赖真实后端、PostgreSQL、Ollama 或任何云端 key，测试稳定且不产生任何费用。
 */

export const MOCK_BOOKS = ["雾隐山庄", "《诡秘之主》（精校版全本）作者：爱潜水的乌贼"];

export const MOCK_MODELS = {
  models: ["qwen2.5:7b", "claude:sonnet", "glm:glm-4-flash"],
  current: "qwen2.5:7b",
};

export const MOCK_INDEX_TASK = {
  id: "index-task-1",
  status: "completed",
  stage: "complete",
  progress: 100,
  message: "书架索引已更新",
  error: null,
  force: false,
  retry_of: null,
  result: {
    novels: MOCK_BOOKS,
    chunk_count: 3,
    added: ["新小说"],
    modified: [],
    deleted: [],
    unchanged: MOCK_BOOKS,
    contextualized: 0,
    relations: 0,
  },
  created_at: "2026-08-09T00:00:00Z",
  started_at: "2026-08-09T00:00:00Z",
  finished_at: "2026-08-09T00:00:01Z",
};

export interface MockAskOptions {
  trace?: { step: string; detail: string }[];
  sources?: { novel: string; chunk_id: number; text: string }[];
  tokens?: string[];
}

const DEFAULT_TRACE = [
  { step: "理解问题", detail: "识别到你在问《雾隐山庄》" },
  { step: "检索范围", detail: "只在《雾隐山庄》内检索" },
  { step: "多路召回", detail: "语义召回 3 条 · 关键词召回 0 条" },
  { step: "融合排序", detail: "合并去重后共 3 个候选，取最相关的 2 段作为依据" },
];

// 注意：文本要足够长（在桌面视口宽度下超过一行），否则 antd 的省略号组件判断
// 不需要截断，不会渲染"展开"链接，测试点击展开时就会找不到元素——之前踩过这个坑。
const DEFAULT_SOURCES = [
  {
    novel: "雾隐山庄",
    chunk_id: 0,
    chapter_title: "第一章 风雪来客",
    text: "三个月前旧疾复发，卧床不起，庄里的药材已经快要用尽，正愁没有人能翻山进来采买，眼下唯一的指望，就是能有名医恰好路过此地。",
  },
  {
    novel: "雾隐山庄",
    chunk_id: 1,
    chapter_title: "第二章 蚀骨奇毒",
    text: "沈砚之带着师父的信前往雾隐山庄寻访名医顾长风，恰逢顾长风旧疾复发且庄中药材匮乏，正是雪中送炭的好时机。",
  },
];

const DEFAULT_TOKENS = ["雾隐", "山庄", "的庄主是", "顾长风", "[1]", "。"];

/** 把 SSE 事件序列拼成一段符合后端格式的响应体：event: xxx\ndata: ...\n\n */
function buildSseBody(opts: MockAskOptions): string {
  const trace = opts.trace ?? DEFAULT_TRACE;
  const sources = opts.sources ?? DEFAULT_SOURCES;
  const tokens = opts.tokens ?? DEFAULT_TOKENS;
  let body = "";
  body += `event: trace\ndata: ${JSON.stringify(trace)}\n\n`;
  body += `event: sources\ndata: ${JSON.stringify(sources)}\n\n`;
  for (const t of tokens) {
    body += `event: token\ndata: ${JSON.stringify(t)}\n\n`;
  }
  body += "event: done\ndata: {}\n\n";
  return body;
}

/** 拦截页面加载时用到的所有 /api/* 请求，换成受控的假数据。 */
export async function mockApi(
  page: Page,
  opts: {
    books?: string[];
    models?: typeof MOCK_MODELS;
    ask?: MockAskOptions;
    /** 会话历史：模拟刷新页面后从后端读回的对话 */
    sessionTurns?: unknown[];
  } = {}
) {
  const books = opts.books ?? MOCK_BOOKS;
  const models = opts.models ?? MOCK_MODELS;

  await page.route("**/api/sessions/**", async (route) => {
    await route.fulfill({ json: { turns: opts.sessionTurns ?? [] } });
  });

  await page.route("**/api/books", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { books } });
    } else {
      // 上传：读一下 multipart 里的文件名，回显为"已保存"，方便测试断言
      await route.fulfill({ json: { saved: books, task: MOCK_INDEX_TASK } });
    }
  });

  await page.route("**/api/index-tasks/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith("/current")) {
      await route.fulfill({ json: null });
    } else {
      await route.fulfill({ json: MOCK_INDEX_TASK });
    }
  });

  await page.route("**/api/reindex**", async (route) => {
    await route.fulfill({ json: MOCK_INDEX_TASK });
  });

  await page.route("**/api/models", async (route) => {
    await route.fulfill({ json: models });
  });

  await page.route("**/api/model", async (route) => {
    await route.fulfill({ json: { current: JSON.parse(route.request().postData() ?? "{}").model } });
  });

  await page.route("**/api/ask", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: buildSseBody(opts.ask ?? {}),
    });
  });
}

/**
 * 拦截 /api/ask 成一个长时间挂起的请求：迟迟不返回响应。
 *
 * 用来测「停止」按钮：普通 mock 一次性返回完整 body，请求瞬间结束、
 * 界面立刻收尾，根本来不及点停止。这里让 handler 长时间不 fulfill，
 * 请求就一直 pending，界面稳定停在"生成中"，可以从容点停止按钮。
 *
 * 前端 abort() 之后这个请求会被取消，handler 里的等待随之作废——
 * 这正是真实的中断路径。
 */
export async function mockHangingAsk(page: Page) {
  await page.route("**/api/ask", async () => {
    // 故意不调用 route.fulfill()：请求保持 pending 直到被 abort 或测试结束。
    // 不需要 sleep，handler 不返回就等于挂住。
  });
}
