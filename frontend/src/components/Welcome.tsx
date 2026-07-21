import { Button, Card } from "antd";

const EXAMPLES = [
  "这本书主要讲了什么故事？",
  "故事里的主角是谁？他/她想做什么？",
  "故事发生在什么地方？",
  "结局是怎样的？",
];

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
