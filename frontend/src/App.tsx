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
const CLAUDE_LABELS: Record<string, string> = {
  haiku: "Claude · haiku（最快最省）",
  sonnet: "Claude · sonnet（推荐）",
  opus: "Claude · opus（最强最慢）",
};

function buildModelOptions(models: string[]) {
  return [
    {
      label: "💻 本地（Ollama，完全离线）",
      options: models
        .filter((m) => !m.startsWith(CLAUDE_PREFIX))
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
  ];
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

export default function App() {
  const isDark = usePrefersDark();
  return (
    <ConfigProvider
      theme={{
        algorithm: isDark
          ? antdTheme.darkAlgorithm
          : antdTheme.defaultAlgorithm,
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

  useEffect(() => {
    refreshBooks();
    refreshModels();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
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

  async function ask(question: string) {
    if (busy || !question.trim()) return;
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);

    await askStream(question, topK, {
      onSources: (s: Source[]) => patchLast((m) => ({ ...m, sources: s })),
      onToken: (t) => patchLast((m) => ({ ...m, content: m.content + t })),
      onDone: () => {
        patchLast((m) => ({ ...m, streaming: false }));
        setBusy(false);
      },
      onError: (e) => {
        patchLast((m) => ({
          ...m,
          streaming: false,
          content: m.content || `⚠️ ${e.message}`,
        }));
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

  const isCloud = currentModel.startsWith(CLAUDE_PREFIX);

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
              <Tooltip title="检索到的原文片段和你的问题会发送到 Anthropic 服务器，并计入你 Claude 订阅的用量">
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
              placeholder="问点什么吧，比如：主角最后怎么样了？"
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
