import { memo, useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Avatar, Collapse, Typography } from 'antd';
import type { AgentStep, RetrievalCandidate, Source, TraceStep } from '../api';

/** 毫秒转成人读的时长：1200 → "1.2s"，340 → "340ms" */
function humanMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  trace?: TraceStep[];
  agentSteps?: AgentStep[];
  streaming?: boolean;
  // 用户点了「停止」：内容是不完整的，界面上要明确告知，别让人以为这就是完整答案
  interrupted?: boolean;
}

const AgentRun = memo(function AgentRun({ steps }: { steps: AgentStep[] }) {
  // 单独 memo 的原因和 Sources 一样：steps 数组在一次回答内引用不变，
  // 打字机每帧刷新气泡时这个工具循环面板不会跟着重渲染。
  if (!steps.length) return null;
  return (
    <Collapse
      className="agent-run-panel"
      size="small"
      defaultActiveKey={['agent']}
      items={[
        {
          key: 'agent',
          label: `🧪 Agent Lab · ${steps.length} 步`,
          children: (
            <ol className="agent-run-steps">
              {steps.map((step) => (
                <li key={`${step.step}-${step.tool}`}>
                  <div className="agent-run-action">
                    <span className="agent-run-index">{step.step}</span>
                    <code>{step.tool}</code>
                    {step.source_ids.length > 0 && <span>{step.source_ids.join(' · ')}</span>}
                  </div>
                  <div className="agent-run-reason">选择：{step.reason}</div>
                  <div className="agent-run-observation">观察：{step.observation}</div>
                </li>
              ))}
            </ol>
          ),
        },
      ]}
    />
  );
});

// 「思考过程」折叠面板：展示检索流水线每一步的真实动作。
//
// 交互参考成熟 AI 应用（ChatGPT / Claude / Perplexity）的三条惯例：
//   1. 检索中展开，让用户看见系统在干什么；**答案开始输出后自动收起**，
//      把注意力还给答案本身（用户手动点过就尊重他的选择，不再自动收）
//   2. 收起时标题给一句有信息量的总结（"思考 1.4s · 5 步"），
//      而不是干巴巴的"5 步"——收起状态才是大多数时候看到的状态
//   3. 步骤逐条点亮，末尾留一个"进行中"的占位，让人知道还没完
const Thinking = memo(function Thinking({ trace, live }: { trace: TraceStep[]; live: boolean }) {
  const [activeKey, setActiveKey] = useState<string[]>(live ? ['t'] : []);
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
          key: 't',
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
                  {totalMs > 0 ? `${humanMs(totalMs)} · ` : ''}
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
  return novel.length > 12 ? novel.slice(0, 12) + '…' : novel;
}

/** 分数展示：BM25 等大分数量级用 1 位小数，向量相似度（-1~1）保留 4 位才有区分度。 */
export function formatScore(candidate: RetrievalCandidate): string {
  if (candidate.score == null) return '—';
  return Math.abs(candidate.score) >= 100 ? candidate.score.toFixed(1) : candidate.score.toFixed(4);
}

/** 检索评测面板：同一个片段在各阶段的 rank 变化，比单看最终答案更能定位问题。 */
const RetrievalEvaluation = memo(function RetrievalEvaluation({ trace }: { trace: TraceStep[] }) {
  const stages = trace.filter((step) => (step.candidates?.length ?? 0) > 0);
  if (!stages.length) return null;
  return (
    <Collapse
      className="retrieval-eval-panel"
      size="small"
      items={[
        {
          key: 'evaluation',
          label: `📊 检索评测 · ${stages.length} 个排名阶段`,
          children: (
            <div className="retrieval-eval-stages">
              {stages.map((stage, stageIndex) => (
                <section className="retrieval-eval-stage" key={`${stage.stage_key}-${stageIndex}`}>
                  <div className="retrieval-eval-title">
                    <strong>{stage.step}</strong>
                    {stage.ms != null && stage.ms > 0 && <span>{humanMs(stage.ms)}</span>}
                  </div>
                  <div className="retrieval-eval-table-wrap">
                    <table className="retrieval-eval-table">
                      <thead>
                        <tr>
                          <th>名次</th>
                          <th>原文位置</th>
                          <th>{stage.candidates?.[0]?.score_label || '分数'}</th>
                          <th>上一阶段</th>
                        </tr>
                      </thead>
                      <tbody>
                        {stage.candidates!.map((candidate) => (
                          <tr
                            className={candidate.selected ? 'candidate-selected' : ''}
                            key={`${candidate.novel}:${candidate.chunk_id}`}
                          >
                            <td>#{candidate.rank}</td>
                            <td title={candidate.chapter_title || candidate.novel}>
                              《{shortName(candidate.novel)}》#{candidate.chunk_id}
                            </td>
                            <td>{formatScore(candidate)}</td>
                            <td>
                              {candidate.previous_rank == null
                                ? '—'
                                : candidate.previous_rank === candidate.rank
                                  ? `#${candidate.previous_rank}`
                                  : `#${candidate.previous_rank} → #${candidate.rank}`}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              ))}
            </div>
          ),
        },
      ]}
    />
  );
});

// 出处卡片单独 memo：打字机每帧都会刷新气泡内容，但 sources 引用在一次回答里不变，
// 所以这一块（含 5 个会做 DOM 测量的省略号组件）在流式输出期间不会重渲染。
const Sources = memo(function Sources({
  sources,
  groupId,
  activeIndex,
}: {
  sources: Source[];
  groupId: string;
  activeIndex: number | null;
}) {
  return (
    <div className="sources-list">
      <div className="sources-label">📖 原文出处</div>
      {sources.map((s, i) => (
        <div
          className={`source-card${activeIndex === i ? ' source-card-active' : ''}`}
          id={`source-${groupId}-${i + 1}`}
          key={i}
        >
          <span className="source-index">{i + 1}</span>
          <span className="source-book">《{shortName(s.novel)}》</span>
          {s.chapter_title && <span className="source-chapter">{s.chapter_title}</span>}
          <Typography.Paragraph
            className="source-text"
            style={{ marginBottom: 0 }}
            ellipsis={{
              rows: 1,
              expandable: 'collapsible',
              symbol: (expanded) => (expanded ? '收起' : '展开'),
            }}
          >
            {s.text}
          </Typography.Paragraph>
        </div>
      ))}
    </div>
  );
});

// 把回答正文里的 [1] [2] 引用替换成可点击按钮，其余文本原样保留。
//
// 用「游标 + 正则 exec 循环」做线性扫描：每个匹配点切一刀，前一段是纯文本、
// 匹配本身变成按钮。流式期间每吐一个字都会重新执行整个解析——但纯字符串
// 切分成本极低，远小于省略号组件的 DOM 测量，不值得为此做增量解析。
function CitedContent({
  text,
  sources,
  onCitation,
}: {
  text: string;
  sources: Source[];
  onCitation: (index: number) => void;
}) {
  const nodes: ReactNode[] = [];
  const citation = /\[(\d+)]/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = citation.exec(text)) !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index));
    const sourceNumber = Number(match[1]);
    if (sourceNumber >= 1 && sourceNumber <= sources.length) {
      nodes.push(
        <button
          type="button"
          className="inline-citation"
          aria-label={`查看原文出处 ${sourceNumber}`}
          onClick={() => onCitation(sourceNumber - 1)}
          key={`${match.index}-${sourceNumber}`}
        >
          [{sourceNumber}]
        </button>,
      );
    } else {
      // 模型偶尔会生成越界编号。保留原文本但不做成可点击按钮，避免把 [9]
      // 错链到第 1 张卡片；离线引用评测会把这种情况记为 invalid。
      nodes.push(match[0]);
    }
    cursor = citation.lastIndex;
  }
  if (cursor < text.length) nodes.push(text.slice(cursor));
  return <>{nodes}</>;
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user';
  // useId 生成每条气泡唯一的出处锚点前缀（替换掉冒号，避免和 DOM id/CSS
  // 选择器的转义规则冲突）；正文 [n] 点击时按它定位到第 n 张出处卡片。
  const groupId = useId().replace(/:/g, '');
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const hasTrace = !isUser && !!msg.trace && msg.trace.length > 0;
  // 「等待正文」：流式中但正文还没到（模型推理/检索中）
  const waiting = !!msg.streaming && !msg.content;

  function jumpToSource(index: number) {
    // 先高亮再等一帧滚动：rAF 保证高亮 class 已经渲染进 DOM 后才定位，
    // 否则可能滚到旧布局的位置上。
    setActiveSource(index);
    requestAnimationFrame(() => {
      document
        .getElementById(`source-${groupId}-${index + 1}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }

  return (
    <div className={`row ${isUser ? 'row-user' : 'row-bot'}`}>
      <Avatar className="avatar" size={36}>
        {isUser ? '🧑' : '📖'}
      </Avatar>
      <div className="bubble">
        {!isUser && hasTrace && <Thinking trace={msg.trace!} live={waiting} />}
        {!isUser && hasTrace && <RetrievalEvaluation trace={msg.trace!} />}
        {!isUser && msg.agentSteps && msg.agentSteps.length > 0 && (
          <AgentRun steps={msg.agentSteps} />
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
            {isUser ? (
              msg.content
            ) : (
              <CitedContent
                text={msg.content}
                sources={msg.sources ?? []}
                onCitation={jumpToSource}
              />
            )}
            {msg.streaming && <span className="caret" />}
            {msg.interrupted && <span className="interrupted-tag">已停止生成</span>}
          </div>
        )}

        {msg.sources && msg.sources.length > 0 && (
          <Sources sources={msg.sources} groupId={groupId} activeIndex={activeSource} />
        )}
      </div>
    </div>
  );
}

// 整个气泡 memo：patchLast 只替换数组最后一项，其余轮次的 msg 引用不变，
// 于是流式打字时旧气泡不会跟着一起重渲染。
export default memo(MessageBubble);
