import { Button, Collapse, List, Slider } from "antd";

interface Props {
  books: string[];
  topK: number;
  busy: boolean;
  onDelete: (name: string) => void;
  onReindex: () => void;
  onClear: () => void;
  setTopK: (n: number) => void;
}

export default function Sidebar({
  books,
  topK,
  busy,
  onDelete,
  onReindex,
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
                  🔄 重新整理书架
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
