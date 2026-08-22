import { expect, test } from '@playwright/test';
import { MOCK_INDEX_TASK, mockApi } from './mock-api';

test.describe('后台索引任务', () => {
  test('刷新后恢复进度，并可以安全停止', async ({ page }) => {
    let cancelled = false;
    const running = {
      ...MOCK_INDEX_TASK,
      status: 'running',
      stage: 'embedding',
      progress: 42,
      message: '《雾隐山庄》Embedding 42/100',
      result: null,
      finished_at: null,
    };
    const stopped = {
      ...running,
      status: 'cancelled',
      stage: 'cancelled',
      message: '任务已取消；当前书没有写入半套索引',
      finished_at: '2026-08-09T00:00:02Z',
    };
    await mockApi(page);
    await page.route('**/api/index-tasks/**', async (route) => {
      if (route.request().method() === 'POST' && route.request().url().endsWith('/cancel')) {
        cancelled = true;
      }
      await route.fulfill({ json: cancelled ? stopped : running });
    });

    await page.goto('/');
    const panel = page.getByLabel('索引任务进度');
    await expect(panel).toContainText('Embedding 42/100');
    await expect(panel).toContainText('42%');
    await panel.getByRole('button', { name: '安全停止' }).click();
    await expect(panel).toContainText('已取消');
    await expect(panel.getByRole('button', { name: '重试未完成内容' })).toBeVisible();
  });

  test('失败原因可见，并能重试', async ({ page }) => {
    let retried = false;
    const failed = {
      ...MOCK_INDEX_TASK,
      status: 'failed',
      stage: 'failed',
      progress: 68,
      message: '索引任务失败，可安全重试',
      error: 'PostgreSQL 连接中断',
      result: null,
    };
    const running = {
      ...failed,
      id: 'index-task-2',
      status: 'running',
      stage: 'scan',
      progress: 1,
      message: '正在重新扫描变化文件',
      error: null,
      retry_of: failed.id,
      finished_at: null,
    };
    await mockApi(page);
    await page.route('**/api/index-tasks/**', async (route) => {
      if (route.request().method() === 'POST' && route.request().url().endsWith('/retry')) {
        retried = true;
      }
      await route.fulfill({ json: retried ? running : failed });
    });

    await page.goto('/');
    const panel = page.getByLabel('索引任务进度');
    await expect(panel).toContainText('PostgreSQL 连接中断');
    await panel.getByRole('button', { name: '重试未完成内容' }).click();
    await expect(panel).toContainText('正在重新扫描变化文件');
    await expect(panel).toContainText('处理中');
  });
});
