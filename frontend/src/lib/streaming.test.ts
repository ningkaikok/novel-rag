import { describe, expect, it } from 'vitest';
import {
  CATCH_UP_TICKS,
  MIN_CHARS_PER_TICK,
  applyStreamError,
  finalizeStreamedMessage,
  takeTypewriterBatch,
} from './streaming';
import type { ChatMessage } from '../components/MessageBubble';

function botMsg(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return { role: 'assistant', content: '', streaming: true, ...overrides };
}

describe('takeTypewriterBatch —— 打字机节奏切分', () => {
  it('积压很短时也至少吐 MIN_CHARS_PER_TICK 个字，一次吐完不剩尾巴', () => {
    const { emit, rest } = takeTypewriterBatch('你好');
    expect(emit).toBe('你好');
    expect(rest).toBe('');
    expect(emit.length).toBeGreaterThanOrEqual(MIN_CHARS_PER_TICK);
  });

  it('单个字的队列也能完整吐出（slice 越界安全截断）', () => {
    const { emit, rest } = takeTypewriterBatch('好');
    expect(emit).toBe('好');
    expect(rest).toBe('');
  });

  it('空队列什么都不吐', () => {
    const { emit, rest } = takeTypewriterBatch('');
    expect(emit).toBe('');
    expect(rest).toBe('');
  });

  it('中等积压按最少字数匀速吐（ceil 未超过下限时不加速）', () => {
    // 长度 2*MIN=4 < 42：ceil(4/42)=1，取 max 后仍是 MIN_CHARS_PER_TICK
    const { emit, rest } = takeTypewriterBatch('一二三四');
    expect(emit).toBe('一二');
    expect(rest).toBe('三四');
  });

  it('大积压时加速追赶：约 CATCH_UP_TICKS 帧内清空', () => {
    const backlog = '字'.repeat(CATCH_UP_TICKS * 100); // 4200 字
    const { emit } = takeTypewriterBatch(backlog);
    expect(emit).toHaveLength(100); // ceil(4200/42)
  });

  it('切分无损：emit + rest 还原原始队列，顺序不变', () => {
    const pending = '林黛玉初进贾府，步步留心，时时在意。';
    const { emit, rest } = takeTypewriterBatch(pending);
    expect(emit + rest).toBe(pending);
  });
});

describe('finalizeStreamedMessage —— 流正常结束的收尾合并', () => {
  it('把队列剩余内容一次性补齐到正文，并结束 streaming 态', () => {
    const msg = botMsg({ content: '贾宝玉是' });
    const out = finalizeStreamedMessage(msg, '《红楼梦》的主角。', false);
    expect(out.content).toBe('贾宝玉是《红楼梦》的主角。');
    expect(out.streaming).toBe(false);
    expect(out.interrupted).toBeFalsy();
  });

  it('用户点过停止时强制标记 interrupted，即使之前不是中断态', () => {
    const out = finalizeStreamedMessage(botMsg({ content: '写到一半…' }), '剩下的字', true);
    expect(out.interrupted).toBe(true);
    expect(out.content).toBe('写到一半…剩下的字');
  });

  it('未停止时保留消息上已有的 interrupted 标记（历史中断不被洗掉）', () => {
    const out = finalizeStreamedMessage(botMsg({ interrupted: true }), '', false);
    expect(out.interrupted).toBe(true);
  });

  it('返回新对象、不改入参：保证 memo 气泡的引用比较语义', () => {
    const msg = botMsg({ content: 'abc' });
    const out = finalizeStreamedMessage(msg, 'def', false);
    expect(out).not.toBe(msg);
    expect(msg.content).toBe('abc');
  });
});

describe('applyStreamError —— 流式请求失败的收尾', () => {
  it('主动停止视为中断：保留已生成内容，不显示 ⚠️ 报错', () => {
    const out = applyStreamError(botMsg({ content: '前半段' }), '后半段', true, 'network down');
    expect(out.content).toBe('前半段后半段');
    expect(out.content).not.toContain('⚠️');
    expect(out.interrupted).toBe(true);
  });

  it('真出错且一个字都没收到时显示 ⚠️ 错误文案', () => {
    const out = applyStreamError(botMsg(), '', false, '后端 5xx');
    expect(out.content).toBe('⚠️ 后端 5xx');
    expect(out.interrupted).toBe(false);
  });

  it('真出错但已有部分内容时保留原文，不用报错覆盖用户读到的字', () => {
    const out = applyStreamError(botMsg({ content: '已经读到的' }), '', false, 'boom');
    expect(out.content).toBe('已经读到的');
    expect(out.interrupted).toBe(false);
  });

  it('错误收尾同样结束 streaming 态并返回新对象', () => {
    const msg = botMsg();
    const out = applyStreamError(msg, '', false, 'x');
    expect(out.streaming).toBe(false);
    expect(out).not.toBe(msg);
  });
});
