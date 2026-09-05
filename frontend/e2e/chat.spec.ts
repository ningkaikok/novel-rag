import { expect, test } from '@playwright/test';
import { mockApi } from './mock-api';

test.describe('首页与欢迎引导', () => {
  test('展示标题、书架和示例问题', async ({ page }) => {
    await mockApi(page);
    await page.goto('/');

    await expect(page.locator('.hero-title')).toHaveText('📖 书虫');
    await expect(page.getByText('想聊聊哪本书？')).toBeVisible();

    // 书架列表来自 mock 的 /api/books
    await expect(page.locator('.book-name').first()).toContainText('雾隐山庄');

    // 四个示例问题按钮都在
    await expect(page.locator('.examples .ant-btn')).toHaveCount(4);
  });
});

test.describe('提问与流式回答', () => {
  test('可以切换自由问答，并把所选模式随同一个输入框提交', async ({ page }) => {
    await mockApi(page);
    await page.goto('/');

    await page.locator('.answer-mode-select').click();
    await page.locator('.ant-select-dropdown:visible').getByText('💬 自由问答').click();
    await expect(page.locator('.composer input')).toHaveAttribute(
      'placeholder',
      '自由提问，不搜索小说书架……',
    );

    const requestPromise = page.waitForRequest(
      (request) => request.url().endsWith('/api/ask') && request.method() === 'POST',
    );
    await page.locator('.composer input').fill('什么是 RAG？');
    await page.locator('.composer .ant-btn-primary').click();
    const request = await requestPromise;

    expect(request.postDataJSON()).toMatchObject({
      question: '什么是 RAG？',
      mode: 'free',
    });
  });

  test('点击示例问题后：思考过程、原文出处、打字机回答都正确渲染', async ({ page }) => {
    await mockApi(page);
    await page.goto('/');

    await page.locator('.examples .ant-btn').first().click();

    // 用户气泡里是刚点的那个问题
    await expect(page.locator('.row-user .content')).toHaveText('这本书主要讲了什么故事？');

    // 打字机效果：Playwright 的断言自带轮询重试，会一直等到逐字输出完成，
    // 不需要手动 sleep，也不会因为 mock 响应到达得快慢而变得脆弱。
    await expect(page.locator('.row-bot .content')).toHaveText('雾隐山庄的庄主是顾长风[1]。', {
      timeout: 10_000,
    });
    // 打完之后光标应当消失、输入框恢复可用
    await expect(page.locator('.caret')).toHaveCount(0);
    await expect(page.locator('.composer input')).toBeEnabled();

    // 思考过程面板：生成结束后标题右侧从跳动的点切换成「N 步」徽章
    const thinking = page.locator('.thinking-panel');
    await expect(thinking.locator('.thinking-panel-label')).toContainText('🔍 思考过程');
    await expect(thinking.locator('.thinking-panel-count')).toContainText('6 步');
    // 回答完成后面板按产品设计自动收起；手动展开后再核对详细步骤。
    await thinking.locator('.ant-collapse-header').click();
    await expect(thinking.locator('.thinking-step-name').nth(0)).toHaveText('理解问题');
    await expect(thinking.locator('.thinking-step-detail').nth(0)).toContainText('《雾隐山庄》');

    // 检索评测面板保留每一阶段的名次和分数，能看见片段 #1 经重排从第 2 升到第 1。
    const evaluation = page.locator('.retrieval-eval-panel');
    await expect(evaluation).toContainText('4 个排名阶段');
    await evaluation.locator('.ant-collapse-header').click();
    await expect(evaluation).toContainText('向量召回');
    await expect(evaluation).toContainText('BM25 召回');
    await expect(evaluation).toContainText('#2 → #1');

    // 原文出处：2 段 mock 数据都渲染成了出处卡片
    const sources = page.locator('.source-card');
    await expect(sources).toHaveCount(2);
    await expect(sources.nth(0).locator('.source-book')).toHaveText('《雾隐山庄》');
    await expect(sources.nth(0).locator('.source-chapter')).toHaveText('第一章 风雪来客');

    // 答案里的 [1] 是真实按钮；点击后会定位并高亮对应的第一张原文卡片。
    await page.getByRole('button', { name: '查看原文出处 1' }).click();
    await expect(sources.nth(0)).toHaveClass(/source-card-active/);
    await expect(sources.nth(1)).not.toHaveClass(/source-card-active/);

    // 出处默认只显示一行，点"展开"后能看到完整原文（mock 文本足够长，必定会被截断）
    const firstSourceText = sources.nth(0).locator('.source-text');
    await expect(firstSourceText.getByText('展开')).toBeVisible();
    await firstSourceText.getByText('展开').click();
    await expect(firstSourceText).toContainText('眼下唯一的指望');
  });

  test('请求出错时：显示错误信息而不是卡在加载中', async ({ page }) => {
    await mockApi(page);
    // 单独覆盖 /api/ask，让它返回失败
    await page.route('**/api/ask', async (route) => {
      await route.fulfill({
        status: 500,
        json: { error: { code: 'index_not_ready', message: '书架为空或索引未建立' } },
      });
    });
    await page.goto('/');

    await page.locator('.composer input').fill('随便问点什么');
    await page.locator('.composer .ant-btn-primary').click();

    await expect(page.locator('.row-bot .content')).toContainText('⚠️');
    // 输入框应该恢复可用，不会一直卡在 busy 状态
    await expect(page.locator('.composer input')).toBeEnabled();
  });
});

test.describe('按需核实引用', () => {
  test('只有被引用的出处才有核实按钮，点击后展示判定和理由', async ({ page }) => {
    await mockApi(page);
    let requestBody: unknown = null;
    await page.route('**/api/citations/verify', async (route) => {
      requestBody = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          label: 'supported',
          reason: '证据里写明庄主是顾长风',
          statement: '雾隐山庄的庄主是顾长风[1]。',
          model: 'glm:glm-4-flash',
        }),
      });
    });

    await page.goto('/');
    await page.locator('.examples .ant-btn').first().click();
    await expect(page.locator('.row-bot .content')).toHaveText('雾隐山庄的庄主是顾长风[1]。', {
      timeout: 10_000,
    });

    // mock 回答只引用了 [1]，所以两张出处卡片里只有第一张能核实——
    // 没被引用的出处没有可核实的句子，显示按钮只会误导。
    const sources = page.locator('.source-card');
    await expect(sources.nth(0).getByRole('button', { name: '核实这条' })).toBeVisible();
    await expect(sources.nth(1).getByRole('button', { name: '核实这条' })).toHaveCount(0);

    await sources.nth(0).getByRole('button', { name: '核实这条' }).click();

    const result = sources.nth(0).locator('.verify-result');
    await expect(result).toHaveClass(/verify-supported/);
    await expect(result).toContainText('原文支持这句话');
    // 理由必须展示出来：判定本身准确率有限，用户要能看着理由自己判断
    await expect(result).toContainText('证据里写明庄主是顾长风');

    // 请求里带的是这条出处的原文，且指明核实第几条引用
    expect(requestBody).toMatchObject({ citation: 1 });
  });
});
