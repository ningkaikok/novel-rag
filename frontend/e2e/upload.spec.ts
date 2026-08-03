import { expect, test } from "@playwright/test";
import { mockApi } from "./mock-api";

test.describe("上传小说", () => {
  test("选择 .txt 文件后触发上传，书架刷新，出现成功提示", async ({ page }) => {
    let uploadedFilenames: string[] = [];
    await mockApi(page);
    // 单独覆盖 /api/books 的 POST：记录收到的文件名，模拟"已保存"
    await page.route("**/api/books", async (route) => {
      if (route.request().method() === "POST") {
        const body = route.request().postDataBuffer()?.toString("utf-8") ?? "";
        uploadedFilenames = [...body.matchAll(/filename="([^"]+)"/g)].map((m) => m[1]);
        await route.fulfill({ json: { saved: ["新小说"], chunk_count: 42 } });
      } else {
        await route.fulfill({ json: { books: ["雾隐山庄"] } });
      }
    });
    await page.goto("/");

    // antd Upload 背后是一个隐藏的 <input type="file">，可以直接对它设置文件
    await page.locator('.composer-toolbar input[type="file"]').setInputFiles({
      name: "新小说.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("这是一本测试用的小说正文。", "utf-8"),
    });

    await expect(page.getByText("📖 已加入书架！")).toBeVisible();
    expect(uploadedFilenames).toContain("新小说.txt");
  });

  test("上传失败时显示错误提示，而不是静默失败", async ({ page }) => {
    await mockApi(page);
    await page.route("**/api/books", async (route) => {
      if (route.request().method() === "POST") {
        await route.fulfill({
          status: 400,
          json: { error: { code: "no_valid_files", message: "没有有效的 .txt 文件" } },
        });
      } else {
        await route.fulfill({ json: { books: [] } });
      }
    });
    await page.goto("/");

    await page.locator('.composer-toolbar input[type="file"]').setInputFiles({
      name: "broken.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("x", "utf-8"),
    });

    await expect(page.getByText("没有有效的 .txt 文件")).toBeVisible();
  });
});
