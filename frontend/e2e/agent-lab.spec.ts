import { expect, test } from '@playwright/test';
import { mockApi } from './mock-api';

test('Agent Lab 展示工具选择、观察、证据编号和最终引用', async ({ page }) => {
  await mockApi(page);
  await page.goto('/');

  await page.getByLabel('工作模式').first().click();
  await page.getByText('🧪 Agent Lab').click();
  await expect(page.getByRole('combobox', { name: '回答模式' })).toBeDisabled();

  await page.locator('.composer input').fill('顾长风为什么卧床？');
  await page.locator('.composer button').click();

  const panel = page.locator('.agent-run-panel');
  await expect(panel).toContainText('Agent Lab · 2 步');
  await expect(panel).toContainText('search_novels');
  await expect(panel).toContainText('answer_with_citations');
  await expect(panel).toContainText('S1 · S2');
  await expect(panel).toContainText('观察：检索到 2 个相关原文片段');
  await expect(page.locator('.row-bot .content')).toContainText('顾长风[1]');
});
