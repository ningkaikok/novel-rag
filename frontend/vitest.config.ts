import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// 单测配置：独立于 vite.config.ts，开发期的 /api 代理等设置不影响测试。
// jsdom 让涉及 DOM 的用例（如组件渲染）也能跑；当前以纯逻辑单测为主。
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
  },
});
