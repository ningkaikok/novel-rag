import { Alert, Button, Collapse, List, Progress, Slider, Tag } from 'antd';
import type { IndexTask } from '../api';

/**
 * 左侧「我的书架」侧栏，纯展示组件：自己不发请求、不持有任务状态，
 * 只把 App 传下来的书列表和索引任务画出来，用户操作通过回调交回 App 统一处理。
 *
 * 索引任务是本组件的核心：上传/删除/同步在后端都是后台线程，
 * 前端拿到的是同一个 IndexTask 状态机（queued → running → cancelling →
 * completed / failed / cancelled），进度由 App 里 700ms 轮询刷新后传入。
 */
interface Props {
  /** 书架上的全部小说（以磁盘文件为准，不是索引里的）。 */
  books: string[];
  /** 每次问答参考的原文片段数，与后端 TOP_K 同一个值。 */
  topK: number;
  /** App 侧「有事情在跑」的总开关：生成中或有索引任务时禁用增删操作，避免并发写书架。 */
  busy: boolean;
  /** 当前索引任务（含进度、阶段文案、终态结果）；null 表示没有可展示的任务。 */
  indexTask: IndexTask | null;
  onDelete: (name: string) => void;
  onReindex: () => void;
  onCancelIndex: () => void;
  onRetryIndex: () => void;
  onClear: () => void;
  setTopK: (n: number) => void;
}

export default function Sidebar({
  books,
  topK,
  busy,
  indexTask,
  onDelete,
  onReindex,
  onCancelIndex,
  onRetryIndex,
  onClear,
  setTopK,
}: Props) {
  return (
    <div className="sidebar-inner">
      <h2 className="shelf-title">📚 我的书架</h2>

      <List
        size="small"
        className="book-list"
        dataSource={books}
        locale={{ emptyText: '书架还是空的，用输入框上方的「📎 添加小说」开始吧' }}
        renderItem={(b) => (
          <List.Item
            actions={[
              <Button
                key="del"
                type="text"
                size="small"
                danger
                disabled={busy}
                onClick={() => onDelete(b)}
              >
                ✕
              </Button>,
            ]}
          >
            <span className="book-name">📕 {b}</span>
          </List.Item>
        )}
      />

      {indexTask && (
        // 索引任务卡片。整个区块只在有任务时渲染，任务进入终态后仍保留，
        // 让用户能看到「这次同步到底改了什么」（added/modified/deleted 摘要）。
        <section className="index-task" aria-label="索引任务进度">
          <div className="index-task-title">
            <span>书架索引</span>
            {/* 状态标签的颜色映射：终态用绿/红/灰一眼分清，
                queued/running/cancelling 都算"还在跑"，统一蓝色 processing。 */}
            <Tag
              color={
                indexTask.status === 'completed'
                  ? 'success'
                  : indexTask.status === 'failed'
                    ? 'error'
                    : indexTask.status === 'cancelled'
                      ? 'default'
                      : 'processing'
              }
            >
              {indexTask.status === 'completed'
                ? '已完成'
                : indexTask.status === 'failed'
                  ? '失败'
                  : indexTask.status === 'cancelled'
                    ? '已取消'
                    : indexTask.status === 'cancelling'
                      ? '正在停止'
                      : '处理中'}
            </Tag>
          </div>
          {/* 进度条状态同样跟随任务状态机：失败红色、完成绿色，
              排队/执行/停止中显示 antd 的 active 流光动画。 */}
          <Progress
            percent={indexTask.progress}
            size="small"
            status={
              indexTask.status === 'failed'
                ? 'exception'
                : indexTask.status === 'completed'
                  ? 'success'
                  : ['queued', 'running', 'cancelling'].includes(indexTask.status)
                    ? 'active'
                    : 'normal'
            }
          />
          <p className="index-task-message">{indexTask.message}</p>
          {/* 失败原因由后端随任务状态一起返回，展示出来方便判断是文件问题还是服务问题 */}
          {indexTask.error && (
            <Alert type="error" showIcon message="失败原因" description={indexTask.error} />
          )}
          {/* 增量索引的结果摘要：只有终态任务才带 result。
              "保留"是文件没变化、索引直接复用的书——这个数字大说明增量同步在生效。 */}
          {indexTask.result && (
            <p className="index-task-summary">
              新增 {indexTask.result.added.length} · 更新 {indexTask.result.modified.length}
              {' · '}删除 {indexTask.result.deleted.length} · 保留{' '}
              {indexTask.result.unchanged.length}
            </p>
          )}
          {/* 操作按钮由状态机决定：还在跑（queued/running/cancelling）显示「安全停止」；
              cancelling 是点了停止后的过渡态——后端要等当前这本书写完事务才真正停，
              期间按钮转圈禁用，防止重复发送取消请求。
              停在 failed/cancelled 终态时给「重试」，从没完成的书继续，不重做全库。 */}
          {(['queued', 'running', 'cancelling'] as string[]).includes(indexTask.status) ? (
            <Button
              block
              size="small"
              danger
              disabled={indexTask.status === 'cancelling'}
              loading={indexTask.status === 'cancelling'}
              onClick={onCancelIndex}
            >
              安全停止
            </Button>
          ) : ['failed', 'cancelled'].includes(indexTask.status) ? (
            <Button block size="small" onClick={onRetryIndex}>
              重试未完成内容
            </Button>
          ) : null}
        </section>
      )}

      {/* 折叠设置区：topK 滑块即时生效（只影响下一次提问）；
          「检查并同步索引」对应后端的增量扫描——没变化的文件会直接跳过。 */}
      <Collapse
        ghost
        className="settings"
        items={[
          {
            key: '1',
            label: '⚙️ 更多设置',
            children: (
              <>
                <div className="slider-label">
                  每次参考多少段原文：<b>{topK}</b>
                </div>
                <Slider min={1} max={10} value={topK} onChange={setTopK} />
                <Button block disabled={busy} onClick={onReindex} style={{ marginTop: 8 }}>
                  🔄 检查并同步索引
                </Button>
                <Button block onClick={onClear} style={{ marginTop: 8 }}>
                  🗑️ 清空对话
                </Button>
              </>
            ),
          },
        ]}
      />
    </div>
  );
}
