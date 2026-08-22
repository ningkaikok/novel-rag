import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

// 前端入口：把 App 挂到 index.html 的 #root 节点上，并引入全局样式。
createRoot(document.getElementById('root')!).render(
  // StrictMode 只在开发环境生效：会把组件挂载/卸载跑两遍、effect 执行两次，
  // 用来提前暴露副作用没清理干净的问题（生产构建里不会有这些行为）。
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
