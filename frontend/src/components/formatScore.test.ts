import { describe, expect, it } from 'vitest';
import { formatScore } from './MessageBubble';
import type { RetrievalCandidate } from '../api';

function candidate(score: number | null | undefined): RetrievalCandidate {
  return { novel: 'demo', chunk_id: 1, rank: 1, score };
}

describe('formatScore —— 检索分数展示', () => {
  it('无分数（null / undefined）显示占位符 —', () => {
    expect(formatScore(candidate(null))).toBe('—');
    expect(formatScore(candidate(undefined))).toBe('—');
  });

  it('BM25 等大量级分数（≥100）保留 1 位小数，避免表格过宽', () => {
    expect(formatScore(candidate(123.456))).toBe('123.5');
    expect(formatScore(candidate(9876.54))).toBe('9876.5');
  });

  it('量级边界恰好为 100 时按大分数处理', () => {
    expect(formatScore(candidate(100))).toBe('100.0');
  });

  it('略低于阈值的分数仍保留 4 位小数', () => {
    expect(formatScore(candidate(99.99))).toBe('99.9900');
  });

  it('向量相似度（-1~1）保留 4 位小数才有区分度，负数同样适用', () => {
    expect(formatScore(candidate(0.5))).toBe('0.5000');
    expect(formatScore(candidate(0.73214))).toBe('0.7321');
    expect(formatScore(candidate(-0.98765))).toBe('-0.9877');
  });
});
