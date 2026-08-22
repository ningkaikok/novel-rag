import { Button, Card } from 'antd';

// 示例问题：新会话没有任何对话时展示，降低「不知道能问什么」的门槛。
// 用 const 放在组件外，避免每次渲染重建数组。
const EXAMPLES = [
  '这本书主要讲了什么故事？',
  '故事里的主角是谁？他/她想做什么？',
  '故事发生在什么地方？',
  '结局是怎样的？',
];

/** 欢迎页：messages 为空时的占位内容。点示例问题等价于把它填进输入框直接提问。 */
export default function Welcome({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="welcome">
      <Card className="welcome-card" bordered={false}>
        <h3>想聊聊哪本书？</h3>
        <p>直接在下面输入你的问题，或点一个示例问题试试看 👇</p>
      </Card>
      <div className="examples">
        {EXAMPLES.map((q) => (
          <Button key={q} block size="large" onClick={() => onPick(q)}>
            {q}
          </Button>
        ))}
      </div>
    </div>
  );
}
