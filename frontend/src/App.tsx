import { useEffect, useState } from "react";
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
  deleteBook,
  listModels,
  reindex,
  setModel as apiSetModel,
  type AnswerMode,
} from "./api";
import Sidebar from "./components/Sidebar";
import Welcome from "./components/Welcome";
import MessageBubble from "./components/MessageBubble";
import { useBookshelf } from "./hooks/useBookshelf";
import { useChatStream } from "./hooks/useChatStream";

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

  // 书架与后台索引任务：列表、上传/删除/同步入口、进度轮询与终态提示，
  // 状态逻辑见 hooks/useBookshelf.ts。
  const {
    books,
    indexTask,
    indexActive,
    startShelfTask,
    handleUpload,
    cancelCurrentIndex,
    retryCurrentIndex,
  } = useBookshelf();

  // 和后端 config.TOP_K 保持一致。3 是实测出来的：开了重排之后 3/5/10 三档
  // 命中率完全相同，取 3 能少送 39% 的字（见 docs/rag-techniques.md 第 5 节）。
  // 侧栏滑块可以随时调，这里只是默认值。
  const [topK, setTopK] = useState(3);
  const [models, setModels] = useState<string[]>([]);
  const [currentModel, setCurrentModel] = useState("");
  const [answerMode, setAnswerMode] = useState<AnswerMode>("auto");
  // 工作模式和回答模式是两层概念：标准 RAG 内部才有 auto/grounded/free；
  // Agent Lab 走独立端点和轨迹，不能硬塞成第四种 AnswerMode，否则后端路由、
  // 会话历史和 UI 状态会混在一起，初学者也看不清“固定流水线 vs 工具循环”。
  const [workspaceMode, setWorkspaceMode] = useState<"rag" | "agent">("rag");

  // 聊天消息数组、SSE 流式请求、停止/打字机与滚动跟随的状态逻辑，
  // 见 hooks/useChatStream.ts。检索评测面板的展开/定位状态在 MessageBubble
  // 组件内部，trace 数据随消息数组由该 hook 维护。
  const {
    messages,
    setMessages,
    input,
    setInput,
    busy,
    ask,
    stopGenerating,
    scrollRef,
    showJumpToLatest,
    hasNewBelow,
    jumpToLatest,
  } = useChatStream({ topK, answerMode, workspaceMode });

  useEffect(() => {
    refreshModels();
  }, []);

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

  const isCloud = isCloudModel(currentModel);
  const cloudVendor = currentModel.startsWith(GLM_PREFIX) ? "智谱" : "Anthropic";
  // 云端隐私提示文案：自由问答不发送小说原文，只有问题本身出网；
  // 其余模式（含 Agent Lab）都会把召回的片段一起发出去，提示要更醒目。
  const cloudPrivacyText =
    workspaceMode !== "agent" && answerMode === "free"
      ? `你的问题会发送到${cloudVendor}；自由问答不会发送小说原文，并计入你自己的账号用量`
      : `检索到的原文片段和你的问题会发送到${cloudVendor}的服务器，并计入你自己的账号用量`;

  return (
    <Layout className="layout" style={{ minHeight: "100vh" }}>
      <Layout.Sider width={300} theme="light" className="sidebar">
        <Sidebar
          books={books}
          topK={topK}
          busy={busy || indexActive}
          indexTask={indexTask}
          onDelete={(name) =>
            startShelfTask(() => deleteBook(name), `已移除《${name}》，正在清理索引`)
          }
          onReindex={() => startShelfTask(() => reindex(), "正在检查书架变化")}
          onCancelIndex={cancelCurrentIndex}
          onRetryIndex={retryCurrentIndex}
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
              disabled={busy || indexActive}
              beforeUpload={(file, fileList) => {
                // antd 会对选中的每个文件各调一次 beforeUpload。这里只想
                // 整批上传一次：判断「当前是不是最后一个文件」，是才把整批
                // 交给 handleUpload；返回 false 同时阻止 antd 自己发请求。
                if (file === fileList[fileList.length - 1]) {
                  handleUpload(fileList as File[]);
                }
                return false; // 阻止 antd 自行上传
              }}
            >
              <Button size="small" disabled={busy || indexActive}>
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

            <Tooltip
              title={
                workspaceMode === "agent"
                  ? "最多五步调用只读工具，并展示选择、观察和证据"
                  : "固定 RAG 流水线，适合日常问答和检索评测"
              }
            >
              <Select
                aria-label="工作模式"
                className="workspace-mode-select"
                size="small"
                value={workspaceMode}
                disabled={busy}
                onChange={(value: "rag" | "agent") => setWorkspaceMode(value)}
                options={[
                  { value: "rag", label: "📖 标准 RAG" },
                  { value: "agent", label: "🧪 Agent Lab" },
                ]}
                popupMatchSelectWidth={false}
              />
            </Tooltip>

            <Tooltip title={ANSWER_MODE_HELP[answerMode]}>
              <Select
                aria-label="回答模式"
                className="answer-mode-select"
                size="small"
                value={answerMode}
                disabled={busy || workspaceMode === "agent"}
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
                workspaceMode === "agent"
                  ? "让 Agent 选择搜索、读取邻居或章节，再依据原文回答……"
                  : answerMode === "grounded"
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
