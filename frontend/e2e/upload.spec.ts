import { expect, test } from '@playwright/test';
import { mockApi } from './mock-api';

test.describe('上传文档', () => {
  test('选择 .txt 文件后触发通用上传，书架刷新，出现成功提示', async ({ page }) => {
    let uploadedFilenames: string[] = [];
    await mockApi(page);
    // mockApi 已提供通用 POST 响应；这里只监听请求并记录文件名，不再覆盖同一路由。
    page.on('request', (request) => {
      if (request.url().includes('/api/knowledge/documents') && request.method() === 'POST') {
        const body = request.postDataBuffer()?.toString('utf-8') ?? '';
        uploadedFilenames = [...body.matchAll(/filename="([^"]+)"/g)].map((m) => m[1]);
      }
    });
    await page.goto('/');

    // antd Upload 背后是一个隐藏的 <input type="file">，可以直接对它设置文件
    await page.locator('.composer-toolbar input[type="file"]').setInputFiles({
      name: '新小说.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('这是一本测试用的小说正文。', 'utf-8'),
    });

    const progress = page.getByLabel('索引任务进度');
    await expect(progress.getByText('书架索引已更新')).toBeVisible();
    await expect(progress).toContainText('已完成');
    expect(uploadedFilenames).toContain('新小说.txt');
  });

  test('上传失败时显示错误提示，而不是静默失败', async ({ page }) => {
    await mockApi(page);
    await page.unroute('**/api/knowledge/documents**');
    await page.route('**/api/knowledge/documents**', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 400,
          json: { error: { code: 'no_valid_files', message: '没有有效的文档文件' } },
        });
      } else {
        await route.fulfill({ json: { documents: [] } });
      }
    });
    await page.goto('/');

    await page.locator('.composer-toolbar input[type="file"]').setInputFiles({
      name: 'broken.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('x', 'utf-8'),
    });

    await expect(page.getByText('没有有效的文档文件')).toBeVisible();
  });
});
