import { expect, test } from "@playwright/test";
import { mockApi } from "./mock-api";

test.describe("模型切换与隐私提示", () => {
  test("下拉框按来源分组，切到云端模型后隐私胶囊变化", async ({ page }) => {
    await mockApi(page);
    await page.goto("/");

    // 初始是本地模型：胶囊显示"完全本地"
    await expect(page.locator(".privacy-chip")).toHaveClass(/local/);
    await expect(page.locator(".privacy-chip")).toContainText("完全本地");

    // 打开下拉框，确认按来源分组（本地 / Claude / 智谱），且各自选项在其分组下
    await page.locator(".model-select-compact").click();
    const dropdown = page.locator(".ant-select-dropdown:visible");
    await expect(dropdown.getByText("💻 本地（Ollama，完全离线）")).toBeVisible();
    await expect(dropdown.getByText("☁️ 我的 Claude 订阅（云端）")).toBeVisible();
    await expect(dropdown.getByText("☁️ 智谱 GLM（云端）")).toBeVisible();

    // 选择智谱的选项
    await dropdown.getByText("GLM-4-Flash（免费最快）").click();

    // 切换后：胶囊变成云端提示，且文案点名"智谱"而不是 Anthropic
    await expect(page.locator(".privacy-chip")).toHaveClass(/cloud/);
    await expect(page.locator(".privacy-chip")).toContainText("云端");
    await page.locator(".privacy-chip").hover();
    await expect(page.getByRole("tooltip")).toContainText("智谱");
  });

  test("没有可用模型时下拉框禁用，不会崩溃", async ({ page }) => {
    await mockApi(page, { models: { models: [], current: "" } });
    await page.goto("/");

    await expect(page.locator(".model-select-compact")).toHaveClass(/ant-select-disabled/);
  });
});
