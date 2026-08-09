import { useEffect, useRef, useState } from "react";
import {
  App as AntdApp,
  Button,
  ConfigProvider,
  Input,
  Layout,
  Select,
  Tooltip,
  Upload,
  theme as antdTheme,
} from "antd";
import {
  askStream,
  deleteBook,
  listBooks,
  listModels,
  loadSession,
  reindex,
  setModel as apiSetModel,
  uploadBooks,
  type AnswerMode,
  type Source,
} from "./api";
import Sidebar from "./components/Sidebar";
import Welcome from "./components/Welcome";
import MessageBubble, { type ChatMessage } from "./components/MessageBubble";

const CLAUDE_PREFIX = "claude:";
const GLM_PREFIX = "glm:";
// 所有走云端（数据会离开本机）的模型前缀，用于隐私提示
const CLOUD_PREFIXES = [CLAUDE_PREFIX, GLM_PREFIX];

const ANSWER_MODE_OPTIONS = [
  { value: "auto", label: "✨ 自动判断" },
  { value: "grounded", label: "📖 仅依据原文" },
  { value: "free", label: "💬 自由问答" },
] satisfies { value: AnswerMode; label: string }[];

const ANSWER_MODE_HELP: Record<AnswerMode, string> = {
  auto: "开放问题直接回答；涉及书中内容时自动检索原文",
  grounded: "强制搜索书架，只依据召回的小说原文回答",
  free: "不搜索书架，直接使用当前模型的通用能力回答",
};

const CLAUDE_LABELS: Record<string, string> = {
  haiku: "Claude · haiku（最快最省）",
  sonnet: "Claude · sonnet（推荐）",
  opus: "Claude · opus（最强最慢）",
};
const GLM_LABELS: Record<string, string> = {
  "glm-4-flash": "GLM-4-Flash（免费最快）",
  "glm-4.5-air": "GLM-4.5-Air（轻量）",
  "glm-4.5": "GLM-4.5（均衡）",
  "glm-4.6": "GLM-4.6（最强）",
};

function isCloudModel(m: string) {
  return CLOUD_PREFIXES.some((p) => m.startsWith(p));
}

function buildModelOptions(models: string[]) {
  const groups = [
    {
      label: "💻 本地（Ollama，完全离线）",
      options: models
        .filter((m) => !isCloudModel(m))
        .map((m) => ({ value: m, label: m })),
    },
    {
      label: "☁️ 我的 Claude 订阅（云端）",
      options: models
        .filter((m) => m.startsWith(CLAUDE_PREFIX))
        .map((m) => {
          const alias = m.slice(CLAUDE_PREFIX.length);
          return { value: m, label: CLAUDE_LABELS[alias] ?? m };
        }),
    },
    {
      label: "☁️ 智谱 GLM（云端）",
      options: models
        .filter((m) => m.startsWith(GLM_PREFIX))
        .map((m) => {
          const name = m.slice(GLM_PREFIX.length);
          return { value: m, label: GLM_LABELS[name] ?? name };
        }),
    },
  ];
  // 没配 claude CLI / ZHIPU_API_KEY 时对应分组为空，不展示空标题
  return groups.filter((g) => g.options.length > 0);
}

// 打字机节奏：每 TICK_MS 吐一批字符。
// 后端返回粒度不一致（Ollama 逐 token，GLM 常常一两个大 chunk 就是全文），
// 所以统一在前端排队按字输出，视觉上才是匀速打字。
// 用 ~33ms（≈30fps）而非 16ms：打字机不需要 60fps，帧率减半就把重渲染次数砍掉一半，
// 明显降低长回答时的卡顿。
// 离底部超过这个距离（px）就认为用户"翻到别处去了"，显示回到底部的浮动按钮
const JUMP_THRESHOLD = 200;

const TICK_MS = 33;
// 每次至少吐 2 个字；积压越多吐越快，避免生成快时字幕越落越远
const MIN_CHARS_PER_TICK = 2;
const CATCH_UP_TICKS = 42; // 目标：约 42 帧（≈1.4s）内清空当前积压

// 会话 ID 存在 localStorage：刷新页面后还能拿同一个 ID 把历史捞回来。
// 用 sessionStorage 会在关标签页时丢失，用 localStorage 更符合"我的阅读记录"的预期。
const SESSION_KEY = "novel-rag-session-id";

function getOrCreateSessionId(): string {
  try {
    const saved = localStorage.getItem(SESSION_KEY);
    if (saved) return saved;
    const fresh = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, fresh);
    return fresh;
  } catch {
    // 隐私模式等禁用 storage 的场景：退化成一次性会话，不落库也不报错
    return crypto.randomUUID();
  }
}

function usePrefersDark() {
  const [isDark, setIsDark] = useState(
    () =>
      typeof matchMedia !== "undefined" &&
      matchMedia("(prefers-color-scheme: dark)").matches
  );
  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => setIsDark(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return isDark;
}

const SANS =
  '-apple-system, "PingFang SC", "Microsoft YaHei", "Segoe UI", Roboto, sans-serif';

// 主题令牌：墨蓝主色 + 更大的圆角 + 更轻的边框，让 antd 组件（按钮/输入/下拉/滑块）
// 与自定义样式保持同一套"书卷气"观感。浅色墨蓝偏深，深色下换成更亮的墨蓝保证对比度。
function themeTokens(isDark: boolean) {
  return {
    colorPrimary: isDark ? "#8ea3cc" : "#445069",
    colorInfo: isDark ? "#8ea3cc" : "#445069",
    colorLink: isDark ? "#9db1d6" : "#4a5878",
    borderRadius: 10,
    borderRadiusLG: 14,
    fontFamily: SANS,
    controlHeight: 34,
    boxShadowSecondary: isDark
      ? "0 6px 20px rgba(0,0,0,0.5)"
      : "0 6px 20px rgba(56,50,40,0.10)",
  };
}

export default function App() {
  const isDark = usePrefersDark();
  return (
    <ConfigProvider
      theme={{
        algorithm: isDark
          ? antdTheme.darkAlgorithm
          : antdTheme.defaultAlgorithm,
        token: themeTokens(isDark),
      }}
    >
      <AntdApp>
        <Main />
      </AntdApp>
    </ConfigProvider>
  );
}

function Main() {
  const { message } = AntdApp.useApp();
  const [books, setBooks] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  // 和后端 config.TOP_K 保持一致。3 是实测出来的：开了重排之后 3/5/10 三档
  // 命中率完全相同，取 3 能少送 39% 的字（见 docs/rag-techniques.md 第 5 节）。
  // 侧栏滑块可以随时调，这里只是默认值。
  const [topK, setTopK] = useState(3);
  const [busy, setBusy] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [currentModel, setCurrentModel] = useState("");
  const [answerMode, setAnswerMode] = useState<AnswerMode>("auto");
  const scrollRef = useRef<HTMLDivElement>(null);
  // 是否显示「跳到最近回答」浮动按钮：用户往上翻看历史/出处、离底部较远时出现
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  // 离开底部期间下面是否来了新内容：用来把按钮文案从"跳到最近回答"换成"有新回复"，
  // 区分"我自己翻上来的"和"下面真有我没看到的东西"
  const [hasNewBelow, setHasNewBelow] = useState(false);
  // 打字机队列：待输出的字符、定时器、以及"后端已推完"的标记
  const queueRef = useRef("");
  const timerRef = useRef<number | null>(null);
  const streamEndedRef = useRef(false);
  // 当前这次生成的中断句柄；Stop 按钮调它的 abort()
  const abortRef = useRef<AbortController | null>(null);
  // 用户主动中断的标记：用来区分「点了停止」和「真的出错了」，两者提示文案不同
  const userStoppedRef = useRef(false);
  const sessionIdRef = useRef<string>(getOrCreateSessionId());
  // 历史恢复那一次 setMessages 不算"新内容"：只是刷新页面后把旧对话摆出来，
  // 用户还没来得及滚下去而已，不该被当成"有新回复"提示。只需要跳过紧接着的那一次
  // messages 变化检查，不用担心时序——这次赋值和下面 effect 之间没有其他会改 messages
  // 的调用插进来，所以同步置真、在 effect 里读到时必然还是真。
  const skipNextNewBelowRef = useRef(false);
  // 是否「粘」在底部跟随最新内容。
  //
  // **这是按用户意图记的状态，不是每次内容更新去量距离算出来的**——两者的差别
  // 在流式输出时非常致命。之前的写法是「内容更新后测一下距底部还有多远，
  // <120px 才跟随」，问题在于测量发生在内容**已经变长之后**：只要某一帧塞进来
  // 的东西高过阈值（markdown 重新排版、出处卡片一次性渲染出来、思考过程展开），
  // 距离瞬间就窜过阈值，自动跟随**从此永久停住**——而用户根本没滚过屏幕，
  // 只能眼看着答案往下跑，或者去点那个浮动按钮。
  //
  // 成熟的聊天应用（ChatGPT、Claude）都是记意图：**只有用户自己往上滚才解除跟随**，
  // 滚回底部就重新粘上。内容涨得多快都不影响这个状态。
  const pinnedRef = useRef(true);

  useEffect(() => {
    refreshBooks();
    refreshModels();
    restoreHistory();
  }, []);

  // 卸载时清掉定时器，避免在已销毁的组件上 setState
  useEffect(() => () => stopTyping(), []);

  // 自动滚到底：粘住时才跟随（往上翻看出处时不打断），
  // 并用 rAF 合并同一帧内的多次触发，避免打字机每帧都强制重排。
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !pinnedRef.current) return;
    const id = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

  // 用户是不是自己在往上翻。
  //
  // 只认这三种**明确来自用户**的输入，不认 scroll 事件——scroll 分不清是人滚的
  // 还是上面那个自动跟随滚的，拿它判断会自己把自己解除掉。
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    function unpinIfScrollingUp(e: WheelEvent) {
      if (e.deltaY < 0) pinnedRef.current = false;
    }
    let touchY = 0;
    function onTouchStart(e: TouchEvent) {
      touchY = e.touches[0]?.clientY ?? 0;
    }
    function onTouchMove(e: TouchEvent) {
      // 手指往下划 = 内容往上走 = 在看上面的内容
      if ((e.touches[0]?.clientY ?? 0) > touchY + 4) pinnedRef.current = false;
    }
    function onKeyDown(e: KeyboardEvent) {
      if (["PageUp", "ArrowUp", "Home"].includes(e.key)) pinnedRef.current = false;
    }
    el.addEventListener("wheel", unpinIfScrollingUp, { passive: true });
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: true });
    el.addEventListener("keydown", onKeyDown);
    return () => {
      el.removeEventListener("wheel", unpinIfScrollingUp);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  // 监听滚动位置：离底部较远时出现「跳到最近回答」按钮，方便翻看历史后一键回到底部。
  // 回到底部时重新粘上并清掉"有新回复"——人已经看到了，不该继续提示。
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    function handleScroll() {
      const el = scrollRef.current;
      if (!el) return;
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
      setShowJumpToLatest(gap > JUMP_THRESHOLD);
      if (gap < 40) {
        // 自己滚回底部了，恢复跟随
        pinnedRef.current = true;
        setHasNewBelow(false);
      }
    }
    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, []);

  // 内容更新但用户没在底部时，标记"下面有新回复"。
  // 单靠上面的 scroll 监听不够——新回答流入时用户并没有滚动，
  // 不主动检查的话按钮状态和提示文案都不会更新，用户就完全不知道有新内容。
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || messages.length === 0) return;
    const away = el.scrollHeight - el.scrollTop - el.clientHeight > JUMP_THRESHOLD;
    if (!away) return;
    // 离底部远就先显示普通的"跳到最近回答"——历史恢复导致的也算，这是合理的导航提示
    setShowJumpToLatest(true);
    if (skipNextNewBelowRef.current) {
      // 历史恢复：内容是旧的，只是还没滚过去，不算"新"
      skipNextNewBelowRef.current = false;
      return;
    }
    setHasNewBelow(true);
  }, [messages]);

  function jumpToLatest() {
    const el = scrollRef.current;
    if (!el) return;
    // 生成中内容每帧都在变长，smooth 平滑滚动追不上增长速度，
    // 用户点了却到不了底、按钮一直亮着反而更烦。所以生成期间直接跳到底，
    // 到位之后由"贴底就自动跟随"的逻辑接管；空闲时才用平滑滚动。
    el.scrollTo({ top: el.scrollHeight, behavior: busy ? "auto" : "smooth" });
    // 点这个按钮就是"我要回去看最新的"，重新粘住跟随
    pinnedRef.current = true;
    // 点了就算已读；不等滚动动画结束再清，避免按钮在滑动过程中还闪着"有新回复"
    setHasNewBelow(false);
  }

  // 刷新页面后把这个会话之前的对话捞回来。
  // 失败不提示：历史恢复是增强功能，拿不到就当新会话，不该打扰用户。
  async function restoreHistory() {
    try {
      const turns = await loadSession(sessionIdRef.current);
      if (turns.length === 0) return;
      skipNextNewBelowRef.current = true;
      // 历史恢复不算"跟随"——这些是刷新页面前就看过的旧内容，一次性灌入时
      // 不该被当成"粘住底部"而强行拽到最新。用户停在哪就该看到哪，
      // 由他自己决定要不要滚下去看最近的回答。
      pinnedRef.current = false;
      setMessages(
        turns.map((t) => ({
          role: t.role,
          content: t.content,
          sources: t.sources ?? undefined,
          trace: t.trace ?? undefined,
          // 历史消息一定不在流式中；被中断的那轮标出来，让用户知道内容不完整
          streaming: false,
          interrupted: t.status === "interrupted",
        }))
      );
    } catch {
      // 忽略：当作没有历史
    }
  }

  async function refreshBooks() {
    try {
      setBooks(await listBooks());
    } catch {
      setBooks([]);
    }
  }

  async function refreshModels() {
    try {
      const info = await listModels();
      setModels(info.models);
      setCurrentModel(info.current);
    } catch {
      setModels([]);
    }
  }

  async function handleModelChange(m: string) {
    const prev = currentModel;
    setCurrentModel(m); // 乐观更新，切换失败再回滚
    try {
      await apiSetModel(m);
      message.success(`已切换到 ${m}`);
    } catch (e) {
      setCurrentModel(prev);
      message.error((e as Error).message);
    }
  }

  function patchLast(fn: (m: ChatMessage) => ChatMessage) {
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = fn(copy[copy.length - 1]);
      return copy;
    });
  }

  function stopTyping() {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  // 收尾：把剩下的字一次性补齐，收起光标，解锁输入
  function finishTyping() {
    stopTyping();
    const rest = queueRef.current;
    queueRef.current = "";
    // 用户点过停止就标记为已中断——内容不完整，界面上要如实告知。
    // 注意：abort 后 reader.read() 可能直接返回 done 而不抛异常，
    // 于是走的是正常结束路径（onDone → finishTyping）而不是 onError，
    // 所以中断标记必须在这里也处理，不能只放在 onError 里。
    const stopped = userStoppedRef.current;
    patchLast((m) => ({
      ...m,
      content: m.content + rest,
      streaming: false,
      interrupted: stopped || m.interrupted,
    }));
    setBusy(false);
  }

  function ensureTyping() {
    if (timerRef.current !== null) return;
    timerRef.current = window.setInterval(() => {
      const pending = queueRef.current;
      if (!pending) {
        // 队列空了：后端还在生成就继续等，已经推完则收尾
        if (streamEndedRef.current) finishTyping();
        return;
      }
      const n = Math.max(
        MIN_CHARS_PER_TICK,
        Math.ceil(pending.length / CATCH_UP_TICKS)
      );
      queueRef.current = pending.slice(n);
      patchLast((m) => ({ ...m, content: m.content + pending.slice(0, n) }));
    }, TICK_MS);
  }

  /** 用户点「停止」：切断网络层，后端随之停止向上游模型索取 token。
   *
   * 同时立刻收尾界面——不让打字机把积压的字慢慢吐完。用户按了停止就该马上停住，
   * 已经收到的内容一次性补齐显示，不丢内容也不假装还在生成。
   */
  function stopGenerating() {
    if (!busy) return;
    userStoppedRef.current = true;
    abortRef.current?.abort();
    streamEndedRef.current = true;
    finishTyping();
  }

  async function ask(question: string) {
    if (busy || !question.trim()) return;
    setInput("");
    // 注意：这里**不**强制恢复跟随。发问时如果人正翻在历史上面（gap 很大），
    // 新回答应该像任何"下面来了新内容"一样走「有新回复」提示，而不是把人
    // 直接拽回底部——那样反而打断了他正在看的东西。跟随与否仍然只由
    // 用户自己的滚动动作决定（见下面的 wheel/touch/keydown 监听）。
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);
    queueRef.current = "";
    streamEndedRef.current = false;
    userStoppedRef.current = false;
    const controller = new AbortController();
    abortRef.current = controller;
    ensureTyping();

    await askStream(
      question,
      topK,
      {
        // 检索每完成一步就追加一条，思考过程逐条点亮（不是等 2 秒后一次性弹出）
        onStep: (s) =>
          patchLast((m) => ({ ...m, trace: [...(m.trace ?? []), s] })),
        onTrace: (t) => patchLast((m) => ({ ...m, trace: t })),
        onSources: (s: Source[]) => patchLast((m) => ({ ...m, sources: s })),
        // 不直接落到界面上，先进队列，由定时器按字吐出
        onToken: (t) => {
          queueRef.current += t;
          ensureTyping();
        },
        onDone: () => {
          streamEndedRef.current = true; // 队列吐完后由定时器收尾
        },
        onError: (e) => {
          // 不再慢慢打了，把已收到的内容一次补齐
          streamEndedRef.current = true;
          stopTyping();
          const rest = queueRef.current;
          queueRef.current = "";
          // 用户主动停止不是错误：保留已生成的内容、标记为已中断，不显示报错文案。
          // 只有真的失败（网络、后端 5xx）才显示 ⚠️。
          const stopped = userStoppedRef.current || e.name === "AbortError";
          patchLast((m) => {
            const content = m.content + rest;
            return {
              ...m,
              streaming: false,
              interrupted: stopped,
              content: stopped ? content : content || `⚠️ ${e.message}`,
            };
          });
          setBusy(false);
        },
      },
      {
        signal: controller.signal,
        sessionId: sessionIdRef.current,
        mode: answerMode,
      }
    );
  }

  async function withShelfToast(action: () => Promise<void>, done: string) {
    message.open({ key: "shelf", type: "loading", content: "正在整理书架…", duration: 0 });
    try {
      await action();
      await refreshBooks();
      message.open({ key: "shelf", type: "success", content: done, duration: 2 });
    } catch (e) {
      message.open({
        key: "shelf",
        type: "error",
        content: (e as Error).message,
        duration: 3,
      });
    }
  }

  function handleUpload(files: File[]) {
    withShelfToast(() => uploadBooks(files), "📖 已加入书架！");
  }

  const isCloud = isCloudModel(currentModel);
  const cloudVendor = currentModel.startsWith(GLM_PREFIX) ? "智谱" : "Anthropic";
  const cloudPrivacyText =
    answerMode === "free"
      ? `你的问题会发送到${cloudVendor}；自由问答不会发送小说原文，并计入你自己的账号用量`
      : `检索到的原文片段和你的问题会发送到${cloudVendor}的服务器，并计入你自己的账号用量`;

  return (
    <Layout className="layout" style={{ minHeight: "100vh" }}>
      <Layout.Sider width={300} theme="light" className="sidebar">
        <Sidebar
          books={books}
          topK={topK}
          busy={busy}
          onDelete={(name) =>
            withShelfToast(() => deleteBook(name), `已移除《${name}》`)
          }
          onReindex={() => withShelfToast(() => reindex().then(() => {}), "书架整理完成")}
          onClear={() => setMessages([])}
          setTopK={setTopK}
        />
      </Layout.Sider>

      <Layout.Content className="main-wrap">
        <main className="main">
          <header className="hero">
            <h1 className="hero-title">📖 书虫</h1>
            <p className="hero-sub">读过的小说，随时问它 —— 你的私人阅读助手</p>
          </header>

          <div className="chat-wrap">
            <div className="chat" ref={scrollRef}>
              {messages.length === 0 ? (
                <Welcome onPick={ask} />
              ) : (
                messages.map((m, i) => <MessageBubble key={i} msg={m} />)
              )}
            </div>
            {showJumpToLatest && messages.length > 0 && (
              <Button
                className={`jump-to-latest${hasNewBelow ? " has-new" : ""}`}
                shape="round"
                size="small"
                // 有新内容时用主色实心按钮，更显眼；否则只是普通的"回到底部"
                type={hasNewBelow ? "primary" : "default"}
                onClick={jumpToLatest}
              >
                {hasNewBelow ? "↓ 有新回复" : "↓ 跳到最近回答"}
              </Button>
            )}
          </div>

          <div className="composer-toolbar">
            <Upload
              accept=".txt"
              multiple
              showUploadList={false}
              disabled={busy}
              beforeUpload={(file, fileList) => {
                if (file === fileList[fileList.length - 1]) {
                  handleUpload(fileList as File[]);
                }
                return false; // 阻止 antd 自行上传
              }}
            >
              <Button size="small" disabled={busy}>
                📎 添加小说
              </Button>
            </Upload>

            <Select
              className="model-select-compact"
              size="small"
              value={models.includes(currentModel) ? currentModel : undefined}
              placeholder={models.length ? "选择模型" : "无可用模型"}
              disabled={busy || models.length === 0}
              onChange={handleModelChange}
              options={buildModelOptions(models)}
              popupMatchSelectWidth={false}
            />

            <Tooltip title={ANSWER_MODE_HELP[answerMode]}>
              <Select
                aria-label="回答模式"
                className="answer-mode-select"
                size="small"
                value={answerMode}
                disabled={busy}
                onChange={(value: AnswerMode) => setAnswerMode(value)}
                options={ANSWER_MODE_OPTIONS}
                popupMatchSelectWidth={false}
              />
            </Tooltip>

            {isCloud ? (
              <Tooltip title={cloudPrivacyText}>
                <span className="privacy-chip cloud">☁️ 云端 · 会上传</span>
              </Tooltip>
            ) : (
              <Tooltip title="回答完全在本机 Ollama 生成，书和提问都不会离开你的电脑">
                <span className="privacy-chip local">🔒 完全本地</span>
              </Tooltip>
            )}
          </div>

          <div className="composer">
            <Input
              size="large"
              placeholder={
                answerMode === "grounded"
                  ? "询问人物、剧情或原句，只依据书架原文回答……"
                  : answerMode === "free"
                    ? "自由提问，不搜索小说书架……"
                    : "问小说内容或开放问题，自动选择回答方式……"
              }
              value={input}
              disabled={busy}
              onChange={(e) => setInput(e.target.value)}
              onPressEnter={() => ask(input)}
            />
            {busy ? (
              // 生成中把发送键换成停止键：切断网络层，后端随之停止索取 token
              <Button
                size="large"
                danger
                className="stop-button"
                onClick={stopGenerating}
              >
                ■ 停止
              </Button>
            ) : (
              <Button
                type="primary"
                size="large"
                disabled={!input.trim()}
                onClick={() => ask(input)}
              >
                发送
              </Button>
            )}
          </div>
          <p className="footer-note">book worm · 基于你本地的小说，答案有据可查</p>
        </main>
      </Layout.Content>
    </Layout>
  );
}
