import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 开发期把 /api 代理到 FastAPI 后端，避免 CORS 与硬编码地址
export default defineConfig({
  plugins: [react()],
  server: {
    // 用一个不常见的端口：5173 是 Vite 的默认端口，本机常有别的项目（甚至
    // 别的框架，如 Electron 应用）也在用，容易冲突。
    port: 45173,
    proxy: {
      // 用 127.0.0.1 而非 localhost：Node 18+ 把 localhost 优先解析为 IPv6 ::1，
      // 而 uvicorn 默认只监听 IPv4，会导致代理 ECONNREFUSED。
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
