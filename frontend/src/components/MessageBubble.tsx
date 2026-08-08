import { memo, useEffect, useRef, useState } from "react";
import { Avatar, Collapse, Typography } from "antd";
import type { Source, TraceStep } from "../api";

/** 毫秒转成人读的时长：1200 → "1.2s"，340 → "340ms" */
function humanMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  trace?: TraceStep[];
  streaming?: boolean;
  // 用户点了「停止」：内容是不完整的，界面上要明确告知，别让人以为这就是完整答案
  interrupted?: boolean;
}

// 「思考过程」折叠面板：展示检索流水线每一步的真实动作。
//
// 交互参考成熟 AI 应用（ChatGPT / Claude / Perplexity）的三条惯例：
//   1. 检索中展开，让用户看见系统在干什么；**答案开始输出后自动收起**，
//      把注意力还给答案本身（用户手动点过就尊重他的选择，不再自动收）
//   2. 收起时标题给一句有信息量的总结（"思考 1.4s · 5 步"），
//      而不是干巴巴的"5 步"——收起状态才是大多数时候看到的状态
//   3. 步骤逐条点亮，末尾留一个"进行中"的占位，让人知道还没完
const Thinking = memo(function Thinking({
  trace,
  live,
}: {
  trace: TraceStep[];
  live: boolean;
}) {
  const [activeKey, setActiveKey] = useState<string[]>(live ? ["t"] : []);
  // 用户手动点过展开/收起之后，就不再自动替他做决定
  const touchedRef = useRef(false);

  useEffect(() => {
    if (!live && !touchedRef.current) setActiveKey([]);
  }, [live]);

  const totalMs = trace.reduce((sum, s) => sum + (s.ms ?? 0), 0);

  return (
    <Collapse
      className="thinking-panel"
      ghost
      size="small"
      activeKey={activeKey}
      onChange={(keys) => {
        touchedRef.current = true;
        setActiveKey(keys as string[]);
      }}
      items={[
        {
          key: "t",
          label: (
            <span className="thinking-panel-label">
              🔍 思考过程
              {live ? (
                // 还在检索/生成：标题右侧显示跳动的点
                <span className="thinking-dots" aria-label="生成中">
                  <span className="thinking-dot" />
                  <span className="thinking-dot" />
                  <span className="thinking-dot" />
                </span>
              ) : (
                <span className="thinking-panel-count">
                  {totalMs > 0 ? `${humanMs(totalMs)} · ` : ""}
                  {trace.length} 步
                </span>
              )}
            </span>
          ),
          children: (
            <ol className="thinking-steps">
              {trace.map((s, i) => (
                <li className="thinking-step" key={i}>
                  <span className="thinking-step-name">{s.step}</span>
                  {/* 耗时紧跟阶段名：放在末尾的话，detail 换行后位置会飘。
                      平时不抢注意力，想看"慢在哪"时一眼能找到（精排通常占大头）。 */}
                  {s.ms != null && s.ms > 0 && (
                    <span className="thinking-step-ms">{humanMs(s.ms)}</span>
                  )}
                  <span className="thinking-step-detail">{s.detail}</span>
                </li>
              ))}
              {live && (
                // 进行中的占位：没有它的话，最后一步显示完就像是已经结束了，
                // 而实际上后面可能还有更慢的一步（精排要 2 秒）没跑完。
                <li className="thinking-step thinking-step-pending">
                  <span className="thinking-step-name">
                    <span className="thinking-dots">
                      <span className="thinking-dot" />
                      <span className="thinking-dot" />
                      <span className="thinking-dot" />
                    </span>
                  </span>
                </li>
              )}
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
  const hasTrace = !isUser && !!msg.trace && msg.trace.length > 0;
  // 「等待正文」：流式中但正文还没到（模型推理/检索中）
  const waiting = !!msg.streaming && !msg.content;
  return (
    <div className={`row ${isUser ? "row-user" : "row-bot"}`}>
      <Avatar className="avatar" size={36}>
        {isUser ? "🧑" : "📖"}
      </Avatar>
      <div className="bubble">
        {!isUser && hasTrace && (
          <Thinking trace={msg.trace!} live={waiting} />
        )}
        {waiting ? (
          // 正文还没到。trace 已经在的话，「生成中」由思考过程面板标题里的动画点表示，
          // 这里不再重复；只有 trace 还没到的那一瞬（检索中）才显示独立的思考指示，避免空气泡。
          !hasTrace && (
            <div className="thinking" aria-live="polite">
              <span className="thinking-dots">
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span className="thinking-dot" />
              </span>
              <span className="thinking-label">正在翻书思考…</span>
            </div>
          )
        ) : (
          <div className="content">
            {msg.content}
            {msg.streaming && <span className="caret" />}
            {msg.interrupted && (
              <span className="interrupted-tag">已停止生成</span>
            )}
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
