// ESLint 9 flat config。规则从推荐集起步：
// - no-explicit-any 降为 warn：学习项目允许渐进收紧；
// - react-hooks/exhaustive-deps 为 warn，rules-of-hooks 保持 error；
// - no-empty 允许空 catch（本仓库多处「失败即忽略」的增强逻辑）。
import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  // 构建产物与测试报告不参与 lint
  { ignores: ['dist', 'node_modules', 'playwright-report', 'test-results'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // TS 项目里交给 tsc --noEmit 检查未定义标识符（no-undef 对浏览器全局误报）
      'no-undef': 'off',
      'no-empty': ['error', { allowEmptyCatch: true }],
      '@typescript-eslint/no-explicit-any': 'warn',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'react-refresh/only-export-components': 'warn',
    },
  },
);
