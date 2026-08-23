import { expect, test } from '@playwright/test';
import { mockApi } from './mock-api';
import type { GraphEdge } from '../src/api';

/**
 * 人物关系审核面板（M4）的 e2e：所有 /api/graph/* 请求在这里拦截，
 * 不依赖真实后端。验证面板的展开加载、字段渲染和 通过/拒绝 两条动作。
 */

// mock 直接对齐前端类型契约（satisfies 保证与 api-generated 漂移时 tsc 报错）
const PENDING_EDGES = [
  {
    novel: '青梧镇异闻',
    person_a: '小顺',
    person_b: '沈砚秋',
    relation: '师徒',
    weight: 3,
    direction: '沈砚秋→小顺',
    confidence: 0.85,
    evidence_type: 'explicit',
    source_chunk_ids: [4, 9],
    review_status: 'pending',
    evidence_excerpt: '想学手艺可以，明天来铺子里扫地，工钱抵你欠下的账。',
  },
  {
    novel: '青梧镇异闻',
    person_a: '小顺',
    person_b: '沈砚秋',
    relation: '亲属',
    weight: 5,
    direction: null,
    confidence: 0.3,
    evidence_type: 'co_occurrence',
    source_chunk_ids: [7],
    review_status: 'pending',
    evidence_excerpt: null,
  },
] satisfies GraphEdge[];

test.describe('人物关系审核', () => {
  test('展开面板后加载待审核队列并渲染质量字段', async ({ page }) => {
    await mockApi(page);
    await page.route('**/api/graph/edges**', async (route) => {
      await route.fulfill({
        json: { total: 2, limit: 20, offset: 0, edges: PENDING_EDGES },
      });
    });

    await page.goto('/');
    // 面板默认收起：展开前不应发出任何 /api/graph/edges 请求（懒加载）
    const panel = page.getByText('🔗 人物关系审核');
    await panel.click();

    const firstItem = page.locator('.graph-review-item').first();
    // person_a/person_b 是存储顺序（字典序），方向语义在悬停提示里
    await expect(firstItem).toContainText('小顺 → 沈砚秋');
    await expect(firstItem).toContainText('师徒');
    await expect(firstItem).toContainText('明确陈述');
    // 共现边方向未知：显示占位符而不是箭头
    await expect(page.locator('.graph-review-item').nth(1)).toContainText('同段共现');
    await expect(page.locator('.graph-review-item').nth(1)).toContainText('置信 30%');
  });

  test('通过/拒绝都会写入审核结论并让边离开待办列表', async ({ page }) => {
    await mockApi(page);
    const reviewed: Array<Record<string, unknown>> = [];
    let remaining = [...PENDING_EDGES];
    await page.route('**/api/graph/edges**', async (route) => {
      await route.fulfill({
        json: { total: remaining.length, limit: 20, offset: 0, edges: remaining },
      });
    });
    await page.route('**/api/graph/review', async (route) => {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      reviewed.push(body);
      // 被拒绝的边立即从待审核队列消失（后端可见性过滤的前端投影）
      remaining = remaining.filter(
        (e) =>
          !(e.novel === body.novel && e.person_a === body.person_a && e.relation === body.relation),
      );
      await route.fulfill({ json: { review_status: body.status } });
    });

    await page.goto('/');
    await page.getByText('🔗 人物关系审核').click();

    // 第一条（明确陈述）通过；第二条（共现推断的假边）拒绝。
    // 每条边都有自己的 通过/拒绝 按钮，用条目作用域避免 strict mode 歧义
    await page
      .locator('.ant-list-item')
      .filter({ hasText: '小顺' })
      .first()
      .getByRole('button', { name: '通过' })
      .click();
    await expect(page.locator('.graph-review-item')).toHaveCount(1);
    await page
      .locator('.ant-list-item')
      .filter({ hasText: '小顺' })
      .last()
      .getByRole('button', { name: '拒绝' })
      .click();
    await expect(page.getByText('没有待审核的关系边')).toBeVisible();

    expect(reviewed).toHaveLength(2);
    expect(reviewed[0]).toMatchObject({
      novel: '青梧镇异闻',
      person_a: '小顺',
      person_b: '沈砚秋',
      relation: '师徒',
      status: 'approved',
    });
    expect(reviewed[1]).toMatchObject({ relation: '亲属', status: 'rejected' });
  });

  test('审核提交失败时边回到列表可以重试', async ({ page }) => {
    await mockApi(page);
    await page.route('**/api/graph/edges**', async (route) => {
      await route.fulfill({
        json: { total: 1, limit: 20, offset: 0, edges: [PENDING_EDGES[0]] },
      });
    });
    await page.route('**/api/graph/review', async (route) => {
      await route.fulfill({
        status: 500,
        json: { error: { code: 'internal_error', message: '数据库忙' } },
      });
    });

    await page.goto('/');
    await page.getByText('🔗 人物关系审核').click();
    await page
      .locator('.ant-list-item')
      .filter({ hasText: '小顺' })
      .first()
      .getByRole('button', { name: '通过' })
      .click();

    // 失败 → 乐观移除回滚，这条边还在界面上
    await expect(page.locator('.graph-review-item')).toHaveCount(1);
  });
});
