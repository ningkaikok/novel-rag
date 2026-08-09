import { Alert, Button, Collapse, List, Progress, Slider, Tag } from "antd";
import type { IndexTask } from "../api";

interface Props {
  books: string[];
  topK: number;
  busy: boolean;
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
        locale={{ emptyText: "书架还是空的，用输入框上方的「📎 添加小说」开始吧" }}
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
        <section className="index-task" aria-label="索引任务进度">
          <div className="index-task-title">
            <span>书架索引</span>
            <Tag
              color={
                indexTask.status === "completed"
                  ? "success"
                  : indexTask.status === "failed"
                    ? "error"
                    : indexTask.status === "cancelled"
                      ? "default"
                      : "processing"
              }
            >
              {indexTask.status === "completed"
                ? "已完成"
                : indexTask.status === "failed"
                  ? "失败"
                  : indexTask.status === "cancelled"
                    ? "已取消"
                    : indexTask.status === "cancelling"
                      ? "正在停止"
                      : "处理中"}
            </Tag>
          </div>
          <Progress
            percent={indexTask.progress}
            size="small"
            status={
              indexTask.status === "failed"
                ? "exception"
                : indexTask.status === "completed"
                  ? "success"
                  : ["queued", "running", "cancelling"].includes(indexTask.status)
                    ? "active"
                    : "normal"
            }
          />
          <p className="index-task-message">{indexTask.message}</p>
          {indexTask.error && (
            <Alert
              type="error"
              showIcon
              message="失败原因"
              description={indexTask.error}
            />
          )}
          {indexTask.result && (
            <p className="index-task-summary">
              新增 {indexTask.result.added.length} · 更新 {indexTask.result.modified.length}
              {" · "}删除 {indexTask.result.deleted.length} · 保留 {indexTask.result.unchanged.length}
            </p>
          )}
          {(["queued", "running", "cancelling"] as string[]).includes(
            indexTask.status
          ) ? (
            <Button
              block
              size="small"
              danger
              disabled={indexTask.status === "cancelling"}
              loading={indexTask.status === "cancelling"}
              onClick={onCancelIndex}
            >
              安全停止
            </Button>
          ) : ["failed", "cancelled"].includes(indexTask.status) ? (
            <Button block size="small" onClick={onRetryIndex}>
              重试未完成内容
            </Button>
          ) : null}
        </section>
      )}

      <Collapse
        ghost
        className="settings"
        items={[
          {
            key: "1",
            label: "⚙️ 更多设置",
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
