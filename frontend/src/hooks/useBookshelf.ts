import { useEffect, useRef, useState } from 'react';
import { App as AntdApp } from 'antd';
import {
  cancelIndexTask,
  getCurrentIndexTask,
  getIndexTask,
  listBooks,
  retryIndexTask,
  uploadBooks,
  type IndexTask,
} from '../api';

/**
 * useBookshelf —— 书架与后台索引任务的全部状态管理（从 App.tsx 抽出，逻辑零变更）。
 *
 * 职责：
 * 1. 持有书架文件列表（books），负责刷新（listBooks）；
 * 2. 持有当前后台索引任务（indexTask）：上传 / 删除 / 手动同步都只是「发起」，
 *    立即拿到一个 IndexTask 存进 state；
 * 3. 任务活跃期间用轻量轮询（700ms）拉取进度；进入终态后只提示一次，
 *    并以磁盘文件为准重新同步书架列表；
 * 4. 刷新页面后通过 current 接口找回同一个任务，不丢「现在跑到哪了」。
 */
export function useBookshelf() {
  const { message } = AntdApp.useApp();
  const [books, setBooks] = useState<string[]>([]);
  const [indexTask, setIndexTask] = useState<IndexTask | null>(null);
  const notifiedIndexTerminalRef = useRef('');

  useEffect(() => {
    refreshBooks();
    restoreIndexTask();
  }, []);

  const indexActive =
    // 任务还在排队/执行/停止中就算「活跃」：驱动轮询，也用于禁用书架增删按钮
    // （后端同一时间只允许一个索引任务，避免上传和删除互相踩）。
    indexTask !== null && ['queued', 'running', 'cancelling'].includes(indexTask.status);

  // 后台线程不占住 HTTP 请求，前端用轻量轮询恢复/更新进度。刷新页面后也会先调用
  // current 接口找回同一个任务，所以不会因为页面重载丢掉“现在跑到哪了”。
  useEffect(() => {
    if (!indexTask || !indexActive) return;
    let requesting = false;
    const timer = window.setInterval(async () => {
      if (requesting) return;
      requesting = true;
      try {
        setIndexTask(await getIndexTask(indexTask.id));
      } catch {
        // 临时网络抖动不把任务判成失败；下一轮继续查询后端真实状态。
      } finally {
        requesting = false;
      }
    }, 700);
    return () => window.clearInterval(timer);
  }, [indexTask?.id, indexActive]);

  // 终态只提示一次，并重新读取文件书架。任务结果里的 added/modified/deleted 是索引
  // 变化；书架列表以磁盘文件为准，所以无论成功/失败都重新同步一次界面。
  useEffect(() => {
    if (!indexTask || indexActive) return;
    refreshBooks();
    const notificationKey = `${indexTask.id}:${indexTask.status}`;
    if (notifiedIndexTerminalRef.current === notificationKey) return;
    notifiedIndexTerminalRef.current = notificationKey;
    if (indexTask.status === 'completed') {
      const result = indexTask.result;
      // 契约里 added/modified/deleted 是可选字段，读的时候按可空兜底
      const changed = result
        ? (result.added?.length ?? 0) +
          (result.modified?.length ?? 0) +
          (result.deleted?.length ?? 0)
        : 0;
      message.success(changed ? `书架索引已更新（处理 ${changed} 本）` : '索引已经是最新');
    } else if (indexTask.status === 'cancelled') {
      message.info('索引任务已安全停止，可以稍后重试');
    } else if (indexTask.status === 'failed') {
      message.error(indexTask.error || '索引任务失败，可在侧栏重试');
    }
  }, [indexTask?.id, indexTask?.status, indexActive]);

  async function refreshBooks() {
    try {
      setBooks(await listBooks());
    } catch {
      setBooks([]);
    }
  }

  async function restoreIndexTask() {
    try {
      const task = await getCurrentIndexTask();
      // 已经结束的旧任务只恢复卡片，不在每次刷新页面时重复弹“成功/失败”。
      if (task && !['queued', 'running', 'cancelling'].includes(task.status)) {
        notifiedIndexTerminalRef.current = `${task.id}:${task.status}`;
      }
      setIndexTask(task);
    } catch {
      // 后台任务状态是增强体验，拿不到不影响问答主流程。
    }
  }

  // 上传/删除/手动同步共用这个入口：三者都立刻返回一个 IndexTask，
  // 存进 state 后由上面的轮询 effect 接管进度刷新，这里只负责提示「已开始」。
  async function startShelfTask(action: () => Promise<IndexTask>, started: string) {
    try {
      const task = await action();
      setIndexTask(task);
      await refreshBooks();
      message.info(started);
    } catch (e) {
      message.error((e as Error).message);
    }
  }

  function handleUpload(files: File[]) {
    startShelfTask(() => uploadBooks(files), '小说已保存，正在后台建立增量索引');
  }

  async function cancelCurrentIndex() {
    if (!indexTask) return;
    try {
      setIndexTask(await cancelIndexTask(indexTask.id));
    } catch (e) {
      message.error((e as Error).message);
    }
  }

  async function retryCurrentIndex() {
    if (!indexTask) return;
    try {
      setIndexTask(await retryIndexTask(indexTask.id));
      message.info('已重新扫描变化文件并继续索引');
    } catch (e) {
      message.error((e as Error).message);
    }
  }

  return {
    books,
    indexTask,
    indexActive,
    startShelfTask,
    handleUpload,
    cancelCurrentIndex,
    retryCurrentIndex,
  };
}
