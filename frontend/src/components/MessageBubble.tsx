import { memo } from "react";
import { Avatar, Collapse, Typography } from "antd";
import type { Source, TraceStep } from "../api";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  trace?: TraceStep[];
  streaming?: boolean;
}

// 「思考过程」折叠面板：展示检索流水线每一步的真实动作。
// trace 引用在一次回答里不变，memo 掉避免打字机每帧重渲染。
const Thinking = memo(function Thinking({
  trace,
  live,
}: {
  trace: TraceStep[];
  live: boolean;
}) {
  return (
    <Collapse
      className="thinking-panel"
      ghost
      size="small"
      // 生成中默认展开（让用户看到检索在做什么）；历史消息默认收起。
      // 只作用于初次挂载，之后用户可自行展开/收起。
      defaultActiveKey={live ? ["t"] : []}
      items={[
        {
          key: "t",
          label: (
            <span className="thinking-panel-label">
              🧠 思考过程
              <span className="thinking-panel-count">{trace.length} 步</span>
            </span>
          ),
          children: (
            <ol className="thinking-steps">
              {trace.map((s, i) => (
                <li className="thinking-step" key={i}>
                  <span className="thinking-step-name">{s.step}</span>
                  <span className="thinking-step-detail">{s.detail}</span>
                </li>
              ))}
            </ol>
          ),
        },
      ]}
    />
  );
});

// 从文件名式书名里提取简短标题：优先取《》内的内容，否则截断
function shortName(novel: string): string {
  const m = novel.match(/《([^》]+)》/);
  if (m) return m[1];
  return novel.length > 12 ? novel.slice(0, 12) + "…" : novel;
}

// 出处卡片单独 memo：打字机每帧都会刷新气泡内容，但 sources 引用在一次回答里不变，
// 所以这一块（含 5 个会做 DOM 测量的省略号组件）在流式输出期间不会重渲染。
const Sources = memo(function Sources({ sources }: { sources: Source[] }) {
  return (
    <div className="sources-list">
      <div className="sources-label">📖 原文出处</div>
      {sources.map((s, i) => (
        <div className="source-card" key={i}>
          <span className="source-index">{i + 1}</span>
          <span className="source-book">《{shortName(s.novel)}》</span>
          <Typography.Paragraph
            className="source-text"
            style={{ marginBottom: 0 }}
            ellipsis={{
              rows: 1,
              expandable: "collapsible",
              symbol: (expanded) => (expanded ? "收起" : "展开"),
            }}
          >
            {s.text}
          </Typography.Paragraph>
        </div>
      ))}
    </div>
  );
});

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`row ${isUser ? "row-user" : "row-bot"}`}>
      <Avatar className="avatar" size={36}>
        {isUser ? "🧑" : "📖"}
      </Avatar>
      <div className="bubble">
        {!isUser && msg.trace && msg.trace.length > 0 && (
          <Thinking trace={msg.trace} live={!!msg.streaming && !msg.content} />
        )}
        {msg.streaming && !msg.content ? (
          // 正文还没到（模型思考/检索中）：显示跳动的思考指示，而不是空气泡
          <div className="thinking" aria-live="polite">
            <span className="thinking-dots">
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-dot" />
            </span>
            <span className="thinking-label">正在翻书思考…</span>
          </div>
        ) : (
          <div className="content">
            {msg.content}
            {msg.streaming && <span className="caret" />}
          </div>
        )}

        {msg.sources && msg.sources.length > 0 && (
          <Sources sources={msg.sources} />
        )}
      </div>
    </div>
  );
}

// 整个气泡 memo：patchLast 只替换数组最后一项，其余轮次的 msg 引用不变，
// 于是流式打字时旧气泡不会跟着一起重渲染。
export default memo(MessageBubble);
