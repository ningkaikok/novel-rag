import { expect, test } from "@playwright/test";
import { mockApi, mockHangingAsk } from "./mock-api";

test.describe("中断生成", () => {
  test("生成中出现停止按钮，点了之后保留已生成内容并标记为已停止", async ({ page }) => {
    await mockApi(page);
    // 让 /api/ask 挂住不返回，界面稳定停在"生成中"，才有机会点停止
    await mockHangingAsk(page);
    await page.goto("/");

    await page.locator(".composer input").fill("讲讲韩立的经历");
    await page.locator(".composer .ant-btn-primary").click();

    // 生成中：发送键变成停止键
    const stopButton = page.locator(".stop-button");
    await expect(stopButton).toBeVisible();
    await expect(stopButton).toContainText("停止");

    await stopButton.click();

    // 停止后：出现「已停止生成」标记，光标消失，发送键回来
    await expect(page.locator(".interrupted-tag")).toHaveText("已停止生成");
    await expect(page.locator(".caret")).toHaveCount(0);
    await expect(page.locator(".stop-button")).toHaveCount(0);
    await expect(page.locator(".composer .ant-btn-primary")).toBeVisible();

    // 用户主动停止不是错误，不该显示 ⚠️ 报错文案
    await expect(page.locator(".row-bot .content")).not.toContainText("⚠️");
    // 输入框恢复可用，可以继续提问
    await expect(page.locator(".composer input")).toBeEnabled();
  });

  test("真正出错时仍然显示错误提示（和主动停止区分开）", async ({ page }) => {
    await mockApi(page);
    await page.route("**/api/ask", async (route) => {
      await route.fulfill({
        status: 500,
        json: { error: { code: "internal_error", message: "后端炸了" } },
      });
    });
    await page.goto("/");

    await page.locator(".composer input").fill("随便问问");
    await page.locator(".composer .ant-btn-primary").click();

    // 出错走的是另一条路：显示 ⚠️，且不该被误标成「已停止生成」
    await expect(page.locator(".row-bot .content")).toContainText("⚠️");
    await expect(page.locator(".interrupted-tag")).toHaveCount(0);
  });
});

test.describe("会话历史恢复", () => {
  test("刷新页面后能从后端读回之前的对话", async ({ page }) => {
    await mockApi(page, {
      sessionTurns: [
        { turn_index: 0, role: "user", content: "雾隐山庄的庄主是谁", sources: null, trace: null, status: "complete" },
        {
          turn_index: 1,
          role: "assistant",
          content: "顾长风。",
          sources: [{ novel: "雾隐山庄", chunk_id: 0, text: "顾长风是雾隐山庄的庄主，也是一位名医，三个月前旧疾复发卧床不起。" }],
          trace: [{ step: "理解问题", detail: "识别到你在问《雾隐山庄》" }],
          status: "complete",
        },
      ],
    });
    await page.goto("/");

    // 历史两轮都渲染出来了
    await expect(page.locator(".row-user .content")).toHaveText("雾隐山庄的庄主是谁");
    await expect(page.locator(".row-bot .content")).toContainText("顾长风");
    // 出处和思考过程也一起恢复
    await expect(page.locator(".source-card")).toHaveCount(1);
    await expect(page.locator(".thinking-panel")).toBeVisible();
    // 历史消息不该显示成还在生成中
    await expect(page.locator(".caret")).toHaveCount(0);
  });

  test("被中断的历史轮次恢复后仍标记为已停止", async ({ page }) => {
    await mockApi(page, {
      sessionTurns: [
        { turn_index: 0, role: "user", content: "讲讲结局", sources: null, trace: null, status: "complete" },
        { turn_index: 1, role: "assistant", content: "韩立最后飞升", sources: null, trace: null, status: "interrupted" },
      ],
    });
    await page.goto("/");

    await expect(page.locator(".interrupted-tag")).toHaveText("已停止生成");
  });

  test("没有历史时正常显示欢迎页", async ({ page }) => {
    await mockApi(page, { sessionTurns: [] });
    await page.goto("/");

    await expect(page.getByText("想聊聊哪本书？")).toBeVisible();
    await expect(page.locator(".row")).toHaveCount(0);
  });
});
