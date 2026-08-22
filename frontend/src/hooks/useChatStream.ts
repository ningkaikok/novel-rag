import { useEffect, useRef, useState } from 'react';
import {
  askAgentStream,
  askStream,
  loadSession,
  type AgentStep,
  type AnswerMode,
  type AskHandlers,
  type Source,
} from '../api';
import type { ChatMessage } from '../components/MessageBubble';
// 打字机节奏与消息收尾的纯逻辑抽在 lib/streaming.ts，可独立单测；
// 本 hook 只保留定时器、ref 与 setState 的接线。
import {
  TICK_MS,
  applyStreamError,
  finalizeStreamedMessage,
  takeTypewriterBatch,
} from '../lib/streaming';

/**
 * useChatStream —— 聊天问答的全部状态与副作用（从 App.tsx 抽出，逻辑零变更）。
 *
 * 职责：
 * 1. 持有消息数组、输入框内容、busy 等聊天状态；
 * 2. 消费 api.ts 的 askStream / askAgentStream（SSE 流式请求），
 *    内置打字机队列（按字匀速输出）、停止生成与中断标记；
 * 3. 刷新页面后通过 sessionId 从后端恢复历史会话；
 * 4. 管理「粘底跟随 / 跳到最近回答」的滚动逻辑（scrollRef 供组件挂到聊天容器上）。
 *
 * 关于检索评测面板（useRetrievalTrace）：它的展开/定位状态全部位于
 * components/MessageBubble.tsx 组件内部（Thinking 的展开收起、
 * RetrievalEvaluation 折叠面板、出处卡片定位），不经过本 hook 的 state——
 * 唯一与本 hook 相关的是 trace 数据本身，它随消息数组走，由 onStep/onTrace
 * 写进最后一条消息。因此不单独拆第三个 hook，在此说明。
 */

// 离底部超过这个距离（px）就认为用户"翻到别处去了"，显示回到底部的浮动按钮
const JUMP_THRESHOLD = 200;

// 会话 ID 存在 localStorage：刷新页面后还能拿同一个 ID 把历史捞回来。
// 用 sessionStorage 会在关标签页时丢失，用 localStorage 更符合"我的阅读记录"的预期。
const SESSION_KEY = 'novel-rag-session-id';

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

export interface UseChatStreamOptions {
  /** 检索条数（侧栏滑块），随标准 RAG 请求透传给后端 */
  topK: number;
  /** 回答模式：auto / grounded / free */
  answerMode: AnswerMode;
  /** 工作模式：标准 RAG 或 Agent Lab（决定走哪个流式端点） */
  workspaceMode: 'rag' | 'agent';
}

export function useChatStream({ topK, answerMode, workspaceMode }: UseChatStreamOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  // 是否显示「跳到最近回答」浮动按钮：用户往上翻看历史/出处、离底部较远时出现
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  // 离开底部期间下面是否来了新内容：用来把按钮文案从"跳到最近回答"换成"有新回复"，
  // 区分"我自己翻上来的"和"下面真有我没看到的东西"
  const [hasNewBelow, setHasNewBelow] = useState(false);
  // 打字机队列：待输出的字符、定时器、以及"后端已推完"的标记
  const queueRef = useRef('');
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
      if (['PageUp', 'ArrowUp', 'Home'].includes(e.key)) pinnedRef.current = false;
    }
    el.addEventListener('wheel', unpinIfScrollingUp, { passive: true });
    el.addEventListener('touchstart', onTouchStart, { passive: true });
    el.addEventListener('touchmove', onTouchMove, { passive: true });
    el.addEventListener('keydown', onKeyDown);
    return () => {
      el.removeEventListener('wheel', unpinIfScrollingUp);
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchmove', onTouchMove);
      el.removeEventListener('keydown', onKeyDown);
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
    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
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
    el.scrollTo({ top: el.scrollHeight, behavior: busy ? 'auto' : 'smooth' });
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
          // 只有 Agent Lab 的历史消息才会带这个字段；普通问答恒为 null。
          agentSteps: t.agent_steps ?? undefined,
          // 历史消息一定不在流式中；被中断的那轮标出来，让用户知道内容不完整
          streaming: false,
          interrupted: t.status === 'interrupted',
        })),
      );
    } catch {
      // 忽略：当作没有历史
    }
  }

  // 流式期间更新最后一条消息的唯一入口，是整个打字机性能方案的地基：
  //
  // 1. **不可变更新**：不原地改 msg 对象，而是浅拷贝数组、只替换最后一项。
  //    这样除最后一条外，其余消息的引用都不变——MessageBubble 整体 memo 后，
  //    React 对旧气泡的 props 浅比较直接命中，历史消息完全不重渲染。
  // 2. **为什么只动最后一项**：SSE 的 token/step/sources 只会作用于
  //    刚追加的那条 assistant 消息；前面的轮次已经定型，永远不需要再变。
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
    queueRef.current = '';
    // 用户点过停止就标记为已中断——内容不完整，界面上要如实告知。
    // 注意：abort 后 reader.read() 可能直接返回 done 而不抛异常，
    // 于是走的是正常结束路径（onDone → finishTyping）而不是 onError，
    // 所以中断标记必须在这里也处理，不能只放在 onError 里。
    const stopped = userStoppedRef.current;
    patchLast((m) => finalizeStreamedMessage(m, rest, stopped));
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
      const { emit, rest } = takeTypewriterBatch(pending);
      queueRef.current = rest;
      patchLast((m) => ({ ...m, content: m.content + emit }));
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
    setInput('');
    // 注意：这里**不**强制恢复跟随。发问时如果人正翻在历史上面（gap 很大），
    // 新回答应该像任何"下面来了新内容"一样走「有新回复」提示，而不是把人
    // 直接拽回底部——那样反而打断了他正在看的东西。跟随与否仍然只由
    // 用户自己的滚动动作决定（见下面的 wheel/touch/keydown 监听）。
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question },
      { role: 'assistant', content: '', streaming: true },
    ]);
    setBusy(true);
    queueRef.current = '';
    streamEndedRef.current = false;
    userStoppedRef.current = false;
    const controller = new AbortController();
    abortRef.current = controller;
    ensureTyping();

    const handlers: AskHandlers = {
      // 检索每完成一步就追加一条，思考过程逐条点亮（不是等 2 秒后一次性弹出）
      onStep: (s) => patchLast((m) => ({ ...m, trace: [...(m.trace ?? []), s] })),
      onTrace: (t) => patchLast((m) => ({ ...m, trace: t })),
      onAgentStep: (step: AgentStep) =>
        patchLast((m) => ({
          ...m,
          agentSteps: [...(m.agentSteps ?? []), step],
        })),
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
        queueRef.current = '';
        // 用户主动停止不是错误：保留已生成的内容、标记为已中断，不显示报错文案。
        // 只有真的失败（网络、后端 5xx）才显示 ⚠️。
        const stopped = userStoppedRef.current || e.name === 'AbortError';
        patchLast((m) => applyStreamError(m, rest, stopped, e.message));
        setBusy(false);
      },
    };
    if (workspaceMode === 'agent') {
      await askAgentStream(question, handlers, {
        signal: controller.signal,
        maxSteps: 5,
        sessionId: sessionIdRef.current,
      });
    } else {
      await askStream(question, topK, handlers, {
        signal: controller.signal,
        sessionId: sessionIdRef.current,
        mode: answerMode,
      });
    }
  }

  return {
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
  };
}
