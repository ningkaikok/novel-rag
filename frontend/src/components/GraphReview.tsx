import { memo, useCallback, useEffect, useState } from 'react';
import { Button, Empty, List, Pagination, Tag, Tooltip } from 'antd';
import { listGraphEdges, reviewGraphEdge, type GraphEdge } from '../api';

/**
 * 人物关系审核面板（M4 质量闭环的人工环节）。
 *
 * 建图时共现推断必然产生假边，LLM 抽取也只能降低而不是消灭错误率——
 * 被质量门槛挡住的边不会直接丢弃，而是留在这里逐条人工把关：
 * 「通过」的边作为可信线索参与问答，「拒绝」的边在所有查询里立即消失。
 *
 * 组件刻意自包含：自己拉数据、自己提交审核。Sidebar 的其余部分是纯展示
 * 组件（状态由 App 下发），而审核队列的生命周期只属于这个折叠面板——
 * 没展开就不请求、不占任何全局状态，收进组件内部反而让 App 更干净。
 */
const PAGE_SIZE = 20;

// 证据类型的展示语义：explicit = 片段里有明确关系陈述；co_occurrence = 仅同段出现。
const EVIDENCE_META: Record<string, { label: string; color: string }> = {
  explicit: { label: '明确陈述', color: 'green' },
  co_occurrence: { label: '同段共现', color: 'orange' },
};

function GraphReviewInner() {
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  // 刚处理完的边先从列表里移除、再等下一次拉取对账——界面即时反馈，
  // 不用等网络往返；失败时重新拉一遍就能看到它回到队列。
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const edgeKey = (e: Pick<GraphEdge, 'novel' | 'person_a' | 'person_b' | 'relation'>) =>
    `${e.novel}|${e.person_a}|${e.person_b}|${e.relation}`;

  const refresh = useCallback(async (nextOffset: number) => {
    setLoading(true);
    try {
      const data = await listGraphEdges('pending', PAGE_SIZE, nextOffset);
      setEdges(data.edges ?? []);
      setTotal(data.total ?? 0);
      setDismissed(new Set()); // 新的一页没有"已处理"概念
    } catch {
      // 面板是辅助功能：后端不可用时安静地显示空态，不打扰问答主流程
      setEdges([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh(offset);
  }, [refresh, offset]);

  const handleReview = useCallback(async (edge: GraphEdge, action: 'approved' | 'rejected') => {
    setDismissed((prev) => new Set(prev).add(edgeKey(edge)));
    try {
      await reviewGraphEdge(edge, action);
      setTotal((t) => Math.max(0, t - 1));
    } catch {
      // 提交失败：取消乐观移除，让这条边留在界面上可以重试
      setDismissed((prev) => {
        const next = new Set(prev);
        next.delete(edgeKey(edge));
        return next;
      });
    }
  }, []);

  const visible = edges.filter((e) => !dismissed.has(edgeKey(e)));

  return (
    <div className="graph-review">
      <List
        size="small"
        loading={loading}
        dataSource={visible}
        locale={{
          emptyText: (
            <Empty description="没有待审核的关系边" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          ),
        }}
        renderItem={(edge) => {
          const evidence = EVIDENCE_META[edge.evidence_type ?? ''] ?? {
            label: edge.evidence_type ?? '未知',
            color: 'default',
          };
          return (
            <List.Item
              actions={[
                <Button
                  key="approve"
                  size="small"
                  type="link"
                  onClick={() => void handleReview(edge, 'approved')}
                >
                  通过
                </Button>,
                <Button
                  key="reject"
                  size="small"
                  type="link"
                  danger
                  onClick={() => void handleReview(edge, 'rejected')}
                >
                  拒绝
                </Button>,
              ]}
            >
              <div className="graph-review-item">
                {/* 人物对 + 方向：方向缺失说明这条边来自纯共现，机器判断不出谁指向谁 */}
                <span className="graph-review-pair">
                  {edge.person_a}
                  {edge.direction && edge.direction !== 'none' ? (
                    <Tooltip title={`方向：${edge.direction}`}>
                      <span className="graph-review-arrow"> → </span>
                    </Tooltip>
                  ) : (
                    <span className="graph-review-arrow"> — </span>
                  )}
                  {edge.person_b}
                </span>
                <span className="graph-review-meta">
                  <Tag>{edge.relation}</Tag>
                  <Tag color={evidence.color}>{evidence.label}</Tag>
                  {edge.confidence != null && (
                    // 置信度是模型/统计的自评把握，不是准确率——文案上保持诚实
                    <Tooltip title="抽取时的自评置信度">
                      <Tag>置信 {Math.round((edge.confidence ?? 0) * 100)}%</Tag>
                    </Tooltip>
                  )}
                </span>
                {/* 证据摘录 ≤80 字，帮审核员快速定位；点不进去没关系，
                    来源片段 id 已在悬停提示里 */}
                {edge.evidence_excerpt && (
                  <Tooltip
                    title={`来源片段：#${(edge.source_chunk_ids ?? []).join('、#') || '无'}`}
                  >
                    <span className="graph-review-evidence">「{edge.evidence_excerpt}」</span>
                  </Tooltip>
                )}
              </div>
            </List.Item>
          );
        }}
      />
      {total > PAGE_SIZE && (
        <Pagination
          size="small"
          className="graph-review-pager"
          current={offset / PAGE_SIZE + 1}
          pageSize={PAGE_SIZE}
          total={total}
          onChange={(page) => setOffset((page - 1) * PAGE_SIZE)}
          showSizeChanger={false}
        />
      )}
    </div>
  );
}

// memo：面板收起/展开之外的重渲染（打字机 token 流每帧都在触发）不应
// 连带重算审核列表——props 为空，memo 后它只在自身状态变化时更新。
export default memo(GraphReviewInner);
