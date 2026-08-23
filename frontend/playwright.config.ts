import { defineConfig, devices } from '@playwright/test';

// e2e 测试只测前端渲染逻辑，所有 /api/* 请求都在测试里用 page.route 拦截、
// 返回固定的假数据——不依赖真实后端、PostgreSQL、Ollama 或任何云端 key。
// 这样测试稳定、CI 不用装数据库/模型，也不会消耗任何真实账号额度。
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'github' : 'html',
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // 用生产构建 + 静态预览服务器，而不是 dev server：更快启动、更接近真实产物，
  // 且固定用 4173 端口，不会跟开发者本地正在跑的 `npm run dev`（45173）冲突。
  webServer: {
    command: 'npm run build && npm run preview -- --port 4173',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
