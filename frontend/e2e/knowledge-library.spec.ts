import { expect, test } from '@playwright/test';
import { mockApi } from './mock-api';

test('知识库侧栏展示文档元数据并提供通用上传入口', async ({ page }) => {
  await mockApi(page);
  await page.goto('/');

  await expect(page.getByRole('heading', { name: '📚 知识库' })).toBeVisible();
  await expect(page.getByText('小说', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('已建立索引', { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: '📎 添加文档' })).toBeVisible();
});
