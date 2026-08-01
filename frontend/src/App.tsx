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
  reindex,
  setModel as apiSetModel,
  uploadBooks,
  type Source,
} from "./api";
import Sidebar from "./components/Sidebar";
import Welcome from "./components/Welcome";
import MessageBubble, { type ChatMessage } from "./components/MessageBubble";

const CLAUDE_PREFIX = "claude:";
const GLM_PREFIX = "glm:";
// 所有走云端（数据会离开本机）的模型前缀，用于隐私提示
const CLOUD_PREFIXES = [CLAUDE_PREFIX, GLM_PREFIX];

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
const TICK_MS = 33;
// 每次至少吐 2 个字；积压越多吐越快，避免生成快时字幕越落越远
const MIN_CHARS_PER_TICK = 2;
const CATCH_UP_TICKS = 42; // 目标：约 42 帧（≈1.4s）内清空当前积压

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
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [currentModel, setCurrentModel] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  // 打字机队列：待输出的字符、定时器、以及"后端已推完"的标记
  const queueRef = useRef("");
  const timerRef = useRef<number | null>(null);
  const streamEndedRef = useRef(false);

  useEffect(() => {
    refreshBooks();
    refreshModels();
  }, []);

  // 卸载时清掉定时器，避免在已销毁的组件上 setState
  useEffect(() => () => stopTyping(), []);

  // 自动滚到底：只在用户本就贴着底部时才跟随（往上翻看出处时不打断），
  // 并用 rAF 合并同一帧内的多次触发，避免打字机每帧都强制重排。
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (!nearBottom) return;
    const id = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

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
    patchLast((m) => ({
      ...m,
      content: m.content + rest,
      streaming: false,
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

  async function ask(question: string) {
    if (busy || !question.trim()) return;
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);
    queueRef.current = "";
    streamEndedRef.current = false;
    ensureTyping();

    await askStream(question, topK, {
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
        // 出错就不再慢慢打了，直接把已有内容补齐并显示错误
        streamEndedRef.current = true;
        stopTyping();
        const rest = queueRef.current;
        queueRef.current = "";
        patchLast((m) => {
          const content = m.content + rest;
          return {
            ...m,
            streaming: false,
            content: content || `⚠️ ${e.message}`,
          };
        });
        setBusy(false);
      },
    });
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

          <div className="chat" ref={scrollRef}>
            {messages.length === 0 ? (
              <Welcome onPick={ask} />
            ) : (
              messages.map((m, i) => <MessageBubble key={i} msg={m} />)
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

            {isCloud ? (
              <Tooltip
                title={`检索到的原文片段和你的问题会发送到${cloudVendor}的服务器，并计入你自己的账号用量`}
              >
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
              placeholder="输入问题或关键词，搜索相关原文后让模型回答……"
              value={input}
              disabled={busy}
              onChange={(e) => setInput(e.target.value)}
              onPressEnter={() => ask(input)}
            />
            <Button
              type="primary"
              size="large"
              disabled={busy || !input.trim()}
              loading={busy}
              onClick={() => ask(input)}
            >
              发送
            </Button>
          </div>
          <p className="footer-note">book worm · 基于你本地的小说，答案有据可查</p>
        </main>
      </Layout.Content>
    </Layout>
  );
}
