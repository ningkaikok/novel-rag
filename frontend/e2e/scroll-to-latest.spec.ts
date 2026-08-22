import { expect, test } from '@playwright/test';
import { mockApi } from './mock-api';

// 足够长的历史，保证不管视口多大都会溢出——不依赖具体像素数，只依赖"肯定装不下"
function longTurns(count: number) {
  const filler =
    '这是一段用来撑高内容、确保聊天区域超出视口高度的重复文本，' +
    '不依赖具体像素数字，只要足够长就一定会触发滚动。';
  const turns = [];
  for (let i = 0; i < count; i++) {
    turns.push({
      turn_index: i * 2,
      role: 'user',
      content: `第 ${i + 1} 个问题`,
      sources: null,
      trace: null,
      status: 'complete',
    });
    turns.push({
      turn_index: i * 2 + 1,
      role: 'assistant',
      content: filler.repeat(3),
      sources: null,
      trace: null,
      status: 'complete',
    });
  }
  return turns;
}

test.describe('跳到最近回答 / 有新回复提示', () => {
  test('历史很长且离底部很远时，只显示朴素的「跳到最近回答」，不误判成新回复', async ({ page }) => {
    // 关键区分：历史恢复的是"旧的、之前就看过的"内容，只是刷新页面后还没滚下去，
    // 不该被当成"有新回复"——这是一个真实踩过的坑，之前实现会在这里误报。
    await mockApi(page, { sessionTurns: longTurns(6) });
    await page.goto('/');

    const jumpBtn = page.locator('.jump-to-latest');
    await expect(jumpBtn).toBeVisible();
    await expect(jumpBtn).toHaveText('↓ 跳到最近回答');
    await expect(jumpBtn).not.toHaveClass(/has-new/);
  });

  test('滚开底部后收到新回答会提示「有新回复」，点击后回到底部并清除提示', async ({ page }) => {
    await mockApi(page, { sessionTurns: longTurns(6) });
    await page.goto('/');

    // 历史已经溢出，此时天然停在顶部（restoreHistory 不会自动滚到底）
    const chat = page.locator('.chat');
    await expect
      .poll(() => chat.evaluate((el) => el.scrollHeight - el.clientHeight))
      .toBeGreaterThan(300);

    // 提一个新问题：新内容追加在底部，而人还在顶部——这是真正的"新内容"
    await page.locator('.composer input').fill('再讲讲细节');
    await page.locator('.composer .ant-btn-primary').click();

    // 等生成结束（发送键回来）——不依赖动画中间态，只看最终稳定状态，避免时序脆弱
    await expect(page.locator('.composer .ant-btn-primary')).toBeVisible();

    const jumpBtn = page.locator('.jump-to-latest');
    await expect(jumpBtn).toHaveText('↓ 有新回复');
    await expect(jumpBtn).toHaveClass(/has-new/);

    await jumpBtn.click();

    // 点击后：回到底部附近，提示随之消失（按钮本身在离底部够近时也会隐藏）
    await expect
      .poll(() => chat.evaluate((el) => el.scrollHeight - el.scrollTop - el.clientHeight))
      .toBeLessThan(200);
    await expect(page.locator('.jump-to-latest')).toHaveCount(0);
  });
});
