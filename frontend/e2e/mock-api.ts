import type { Page } from '@playwright/test';
// mock 数据直接对齐前端类型契约（src/api.ts 从 api-generated.ts 取的别名）：
// 用 satisfies 在编辑器/未来把 e2e 纳入 tsc 时就能发现 mock 与后端契约漂移。
import type { AgentStep, IndexTask, ModelsInfo, Source, StoredTurn, TraceStep } from '../src/api';

/**
 * e2e 测试只验证前端渲染逻辑，所有 /api/* 请求都在这里拦截、返回固定假数据。
 * 不依赖真实后端、PostgreSQL、Ollama 或任何云端 key，测试稳定且不产生任何费用。
 */

export const MOCK_BOOKS = ['雾隐山庄', '《诡秘之主》（精校版全本）作者：爱潜水的乌贼'];

export const MOCK_MODELS = {
  models: ['qwen2.5:7b', 'claude:sonnet', 'glm:glm-4-flash'],
  current: 'qwen2.5:7b',
} satisfies ModelsInfo;

export const MOCK_INDEX_TASK = {
  id: 'index-task-1',
  status: 'completed',
  stage: 'complete',
  progress: 100,
  message: '书架索引已更新',
  error: null,
  force: false,
  retry_of: null,
  result: {
    novels: MOCK_BOOKS,
    chunk_count: 3,
    added: ['新小说'],
    modified: [],
    deleted: [],
    unchanged: MOCK_BOOKS,
    contextualized: 0,
    relations: 0,
    hierarchy_nodes: 0,
  },
  created_at: '2026-08-09T00:00:00Z',
  started_at: '2026-08-09T00:00:00Z',
  finished_at: '2026-08-09T00:00:01Z',
} satisfies IndexTask;

// trace / sources 直接复用 REST 契约里的类型：SSE 事件负载虽然不走 OpenAPI，
// 但后端序列化的正是同一批 Pydantic 模型，mock 保持同构才测得出真实回归。
export interface MockAskOptions {
  trace?: TraceStep[];
  sources?: Source[];
  tokens?: string[];
}

// satisfies TraceStep[]：契约里 RetrievalCandidate.selected 是必填
// （后端默认 false，序列化时始终存在），所以前两个阶段的候选要显式补上。
const DEFAULT_TRACE = [
  { step: '理解问题', detail: '识别到你在问《雾隐山庄》' },
  { step: '检索范围', detail: '只在《雾隐山庄》内检索' },
  {
    step: '向量召回',
    detail: '按语义相似度召回 2 个片段',
    ms: 18,
    stage_key: 'vector',
    candidates: [
      {
        novel: '雾隐山庄',
        chunk_id: 1,
        rank: 1,
        score: 0.82,
        score_label: '余弦相似度',
        selected: false,
      },
      {
        novel: '雾隐山庄',
        chunk_id: 0,
        rank: 2,
        score: 0.76,
        score_label: '余弦相似度',
        selected: false,
      },
    ],
  },
  {
    step: 'BM25 召回',
    detail: '按关键词相关性召回 2 个片段',
    ms: 7,
    stage_key: 'bm25',
    candidates: [
      { novel: '雾隐山庄', chunk_id: 0, rank: 1, score: 8.3, score_label: 'BM25', selected: false },
      { novel: '雾隐山庄', chunk_id: 1, rank: 2, score: 4.1, score_label: 'BM25', selected: false },
    ],
  },
  {
    step: '融合排序',
    detail: '合并去重后共 2 个候选',
    ms: 1,
    stage_key: 'rrf',
    candidates: [
      {
        novel: '雾隐山庄',
        chunk_id: 0,
        rank: 1,
        score: 0.0325,
        score_label: 'RRF',
        selected: true,
      },
      {
        novel: '雾隐山庄',
        chunk_id: 1,
        rank: 2,
        score: 0.0325,
        score_label: 'RRF',
        selected: true,
      },
    ],
  },
  {
    step: '精排',
    detail: '取最相关的 2 段',
    ms: 80,
    stage_key: 'rerank',
    candidates: [
      {
        novel: '雾隐山庄',
        chunk_id: 1,
        rank: 1,
        previous_rank: 2,
        score: 0.99,
        score_label: 'CrossEncoder',
        selected: true,
      },
      {
        novel: '雾隐山庄',
        chunk_id: 1,
        rank: 2,
        previous_rank: 1,
        score: 0.91,
        score_label: 'CrossEncoder',
        selected: true,
      },
    ],
  },
] satisfies TraceStep[];

// 注意：文本要足够长（在桌面视口宽度下超过一行），否则 antd 的省略号组件判断
// 不需要截断，不会渲染"展开"链接，测试点击展开时就会找不到元素——之前踩过这个坑。
const DEFAULT_SOURCES = [
  {
    novel: '雾隐山庄',
    chunk_id: 0,
    chapter_title: '第一章 风雪来客',
    text: '三个月前旧疾复发，卧床不起，庄里的药材已经快要用尽，正愁没有人能翻山进来采买，眼下唯一的指望，就是能有名医恰好路过此地。',
  },
  {
    novel: '雾隐山庄',
    chunk_id: 1,
    chapter_title: '第二章 蚀骨奇毒',
    text: '沈砚之带着师父的信前往雾隐山庄寻访名医顾长风，恰逢顾长风旧疾复发且庄中药材匮乏，正是雪中送炭的好时机。',
  },
] satisfies Source[];

const DEFAULT_TOKENS = ['雾隐', '山庄', '的庄主是', '顾长风', '[1]', '。'];

/** 把 SSE 事件序列拼成一段符合后端格式的响应体：event: xxx\ndata: ...\n\n */
function buildSseBody(opts: MockAskOptions): string {
  const trace = opts.trace ?? DEFAULT_TRACE;
  const sources = opts.sources ?? DEFAULT_SOURCES;
  const tokens = opts.tokens ?? DEFAULT_TOKENS;
  let body = '';
  body += `event: trace\ndata: ${JSON.stringify(trace)}\n\n`;
  body += `event: sources\ndata: ${JSON.stringify(sources)}\n\n`;
  for (const t of tokens) {
    body += `event: token\ndata: ${JSON.stringify(t)}\n\n`;
  }
  body += 'event: done\ndata: {}\n\n';
  return body;
}

function buildAgentSseBody(): string {
  const steps = [
    {
      step: 1,
      reason: '先定位与问题相关的原文',
      tool: 'search_novels',
      args: { query: '顾长风为什么卧床' },
      observation: '检索到 2 个相关原文片段',
      source_ids: ['S1', 'S2'],
    },
    {
      step: 2,
      reason: '证据已足够，生成带引用答案',
      tool: 'answer_with_citations',
      args: { source_ids: ['S1', 'S2'] },
      observation: '使用 2 个原文片段生成带引用答案',
      source_ids: ['S1', 'S2'],
    },
  ] satisfies AgentStep[];
  let body = steps.map((step) => `event: agent_step\ndata: ${JSON.stringify(step)}\n\n`).join('');
  body += `event: sources\ndata: ${JSON.stringify(DEFAULT_SOURCES)}\n\n`;
  for (const token of DEFAULT_TOKENS) {
    body += `event: token\ndata: ${JSON.stringify(token)}\n\n`;
  }
  return body + 'event: done\ndata: {}\n\n';
}

/** 拦截页面加载时用到的所有 /api/* 请求，换成受控的假数据。 */
export async function mockApi(
  page: Page,
  opts: {
    books?: string[];
    models?: typeof MOCK_MODELS;
    ask?: MockAskOptions;
    /** 会话历史：模拟刷新页面后从后端读回的对话 */
    sessionTurns?: StoredTurn[];
  } = {},
) {
  const books = opts.books ?? MOCK_BOOKS;
  const models = opts.models ?? MOCK_MODELS;

  await page.route('**/api/sessions/**', async (route) => {
    if (route.request().method() === 'DELETE') {
      await route.fulfill({ json: { session_id: 'mock-session', deleted_turns: 2 } });
    } else {
      await route.fulfill({ json: { turns: opts.sessionTurns ?? [] } });
    }
  });

  await page.route('**/api/books', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({ json: { books } });
    } else {
      // 上传：读一下 multipart 里的文件名，回显为"已保存"，方便测试断言
      await route.fulfill({ json: { saved: books, task: MOCK_INDEX_TASK } });
    }
  });

  await page.route('**/api/index-tasks/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/current')) {
      await route.fulfill({ json: null });
    } else {
      await route.fulfill({ json: MOCK_INDEX_TASK });
    }
  });

  await page.route('**/api/reindex**', async (route) => {
    await route.fulfill({ json: MOCK_INDEX_TASK });
  });

  await page.route('**/api/models', async (route) => {
    await route.fulfill({ json: models });
  });

  await page.route('**/api/model', async (route) => {
    await route.fulfill({
      json: { current: JSON.parse(route.request().postData() ?? '{}').model },
    });
  });

  await page.route('**/api/agent/ask', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: buildAgentSseBody(),
    });
  });

  await page.route('**/api/ask', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
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
  await page.route('**/api/ask', async () => {
    // 故意不调用 route.fulfill()：请求保持 pending 直到被 abort 或测试结束。
    // 不需要 sleep，handler 不返回就等于挂住。
  });
}
