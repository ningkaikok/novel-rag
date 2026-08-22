import type { ChatMessage } from '../components/MessageBubble';

/**
 * 聊天流式输出的纯逻辑层（从 hooks/useChatStream.ts 抽出，行为零变更）。
 *
 * 这里只放「给定输入、返回输出」的函数：打字机的节奏切分、流结束/出错的
 * 消息收尾。它们不碰 React 状态，因此可以在 Vitest 里直接单测；hook 只负责
 * 定时器、ref 和 setState 的接线。
 */

// 打字机节奏：每 TICK_MS 吐一批字符。
// 后端返回粒度不一致（Ollama 逐 token，GLM 常常一两个大 chunk 就是全文），
// 所以统一在前端排队按字输出，视觉上才是匀速打字。
// 用 ~33ms（≈30fps）而非 16ms：打字机不需要 60fps，帧率减半就把重渲染次数砍掉一半，
// 明显降低长回答时的卡顿。
export const TICK_MS = 33;
// 每次至少吐 2 个字；积压越多吐越快，避免生成快时字幕越落越远
export const MIN_CHARS_PER_TICK = 2;
export const CATCH_UP_TICKS = 42; // 目标：约 42 帧（≈1.4s）内清空当前积压

/** 打字机每帧从积压队列里切出一段：返回本轮要吐出的 emit 和剩余的 rest。
 *
 * 切分规则：至少 MIN_CHARS_PER_TICK 个字；积压越大吐越快，
 * 按 ceil(积压长度 / CATCH_UP_TICKS) 计，保证约 CATCH_UP_TICKS 帧内追平。
 * 队列比最少字数还短时一次全部吐出（slice 越界安全截断）。
 */
export function takeTypewriterBatch(pending: string): {
  emit: string;
  rest: string;
} {
  const n = Math.max(MIN_CHARS_PER_TICK, Math.ceil(pending.length / CATCH_UP_TICKS));
  return { emit: pending.slice(0, n), rest: pending.slice(n) };
}

/** 流正常结束（或用户停止后立即收尾）时的消息合并：把队列里剩下的字一次性补齐，
 * 结束 streaming 态；用户点过停止就标记为已中断——内容不完整，界面上要如实告知。
 */
export function finalizeStreamedMessage(
  msg: ChatMessage,
  rest: string,
  stopped: boolean,
): ChatMessage {
  return {
    ...msg,
    content: msg.content + rest,
    streaming: false,
    interrupted: stopped || msg.interrupted,
  };
}

/** 流式请求出错时的消息收尾。
 *
 * 用户主动停止不是错误：保留已生成的内容、标记为已中断，不显示报错文案。
 * 只有真的失败（网络、后端 5xx）且一个字都没收到时，才显示 ⚠️ 错误提示；
 * 已有部分内容则保留原文，避免把用户已经读到的字替换成报错。
 */
export function applyStreamError(
  msg: ChatMessage,
  rest: string,
  stopped: boolean,
  errorMessage: string,
): ChatMessage {
  const content = msg.content + rest;
  return {
    ...msg,
    streaming: false,
    interrupted: stopped,
    content: stopped ? content : content || `⚠️ ${errorMessage}`,
  };
}
