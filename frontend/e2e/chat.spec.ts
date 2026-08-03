import { expect, test } from "@playwright/test";
import { mockApi } from "./mock-api";

test.describe("首页与欢迎引导", () => {
  test("展示标题、书架和示例问题", async ({ page }) => {
    await mockApi(page);
    await page.goto("/");

    await expect(page.locator(".hero-title")).toHaveText("📖 书虫");
    await expect(page.getByText("想聊聊哪本书？")).toBeVisible();

    // 书架列表来自 mock 的 /api/books
    await expect(page.locator(".book-name").first()).toContainText("雾隐山庄");

    // 四个示例问题按钮都在
    await expect(page.locator(".examples .ant-btn")).toHaveCount(4);
  });
});

test.describe("提问与流式回答", () => {
  test("点击示例问题后：思考过程、原文出处、打字机回答都正确渲染", async ({ page }) => {
    await mockApi(page);
    await page.goto("/");

    await page.locator(".examples .ant-btn").first().click();

    // 用户气泡里是刚点的那个问题
    await expect(page.locator(".row-user .content")).toHaveText(
      "这本书主要讲了什么故事？"
    );

    // 打字机效果：Playwright 的断言自带轮询重试，会一直等到逐字输出完成，
    // 不需要手动 sleep，也不会因为 mock 响应到达得快慢而变得脆弱。
    await expect(page.locator(".row-bot .content")).toHaveText(
      "雾隐山庄的庄主是顾长风。",
      { timeout: 10_000 }
    );
    // 打完之后光标应当消失、输入框恢复可用
    await expect(page.locator(".caret")).toHaveCount(0);
    await expect(page.locator(".composer input")).toBeEnabled();

    // 思考过程面板：生成结束后标题右侧从跳动的点切换成「N 步」徽章
    const thinking = page.locator(".thinking-panel");
    await expect(thinking.locator(".thinking-panel-label")).toContainText("🔍 思考过程");
    await expect(thinking.locator(".thinking-panel-count")).toHaveText("4 步");
    await expect(thinking.locator(".thinking-step-name").nth(0)).toHaveText("理解问题");
    await expect(thinking.locator(".thinking-step-detail").nth(0)).toContainText(
      "《雾隐山庄》"
    );

    // 原文出处：2 段 mock 数据都渲染成了出处卡片
    const sources = page.locator(".source-card");
    await expect(sources).toHaveCount(2);
    await expect(sources.nth(0).locator(".source-book")).toHaveText("《雾隐山庄》");

    // 出处默认只显示一行，点"展开"后能看到完整原文（mock 文本足够长，必定会被截断）
    const firstSourceText = sources.nth(0).locator(".source-text");
    await expect(firstSourceText.getByText("展开")).toBeVisible();
    await firstSourceText.getByText("展开").click();
    await expect(firstSourceText).toContainText("眼下唯一的指望");
  });

  test("请求出错时：显示错误信息而不是卡在加载中", async ({ page }) => {
    await mockApi(page);
    // 单独覆盖 /api/ask，让它返回失败
    await page.route("**/api/ask", async (route) => {
      await route.fulfill({
        status: 500,
        json: { error: { code: "index_not_ready", message: "书架为空或索引未建立" } },
      });
    });
    await page.goto("/");

    await page.locator(".composer input").fill("随便问点什么");
    await page.locator(".composer .ant-btn-primary").click();

    await expect(page.locator(".row-bot .content")).toContainText("⚠️");
    // 输入框应该恢复可用，不会一直卡在 busy 状态
    await expect(page.locator(".composer input")).toBeEnabled();
  });
});
