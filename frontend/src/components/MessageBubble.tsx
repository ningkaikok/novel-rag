import { Avatar, Popover } from "antd";
import type { Source } from "../api";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
}

// 从文件名式书名里提取简短标题：优先取《》内的内容，否则截断
function shortName(novel: string): string {
  const m = novel.match(/《([^》]+)》/);
  if (m) return m[1];
  return novel.length > 12 ? novel.slice(0, 12) + "…" : novel;
}

export default function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`row ${isUser ? "row-user" : "row-bot"}`}>
      <Avatar className="avatar" size={36}>
        {isUser ? "🧑" : "📖"}
      </Avatar>
      <div className="bubble">
        <div className="content">
          {msg.content}
          {msg.streaming && <span className="caret" />}
        </div>

        {msg.sources && msg.sources.length > 0 && (
          <div className="sources-links">
            <span className="sources-label">📖 原文出处</span>
            {msg.sources.map((s, i) => (
              <Popover
                key={i}
                trigger="click"
                placement="top"
                title={`出自《${shortName(s.novel)}》`}
                content={<div className="source-popover">{s.text}</div>}
              >
                <a className="source-link">
                  <span className="source-index">{i + 1}</span>
                  {shortName(s.novel)}
                </a>
              </Popover>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
