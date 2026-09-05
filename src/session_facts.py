"""结构化会话事实：这个会话涉及哪本书、出现过哪些人物（M3.6）。

和滚动摘要（session_summary.py）的关系
--------------------------------------
两者都补进「对话背景」，但可信度的来源不同：

- 摘要是模型压缩的产物，可能漂移，默认关闭，需要长会话评测证明没有引入漂移
  才敢打开
- 这里**不调用任何模型**：书名来自这一轮真实检索到的证据本身
  （``turn["sources"]`` 里的 ``novel`` 字段），人物名来自人物关系图里已经确认
  存在的人名反查——要么在数据库里查得到、要么在文本里精确匹配得到，没有
  "大概对"的中间态，因此默认随对话背景一起生效，不需要开关

只覆盖"书名 + 人物"。"用户已确认结论"暂不做：目前唯一能确定信号强度的确认
渠道是引用核实功能（点击"核实这条"），但那只覆盖用户主动触发的极少数情况，
覆盖面太窄；用聊天里的"对/是的/确认了"之类关键词判断"用户确认了什么"，信号
本身就很模糊，容易把"对，但是……"这种反驳误判成确认——比不做还危险。留到
有更可靠的确认信号时再补。
"""

from dataclasses import dataclass, field

from config import SESSION_FACTS_MAX_CHARACTERS
from novel_match import _display_title
from postgres import connect


@dataclass
class SessionFacts:
    # 最近提到的排在最后——"当前小说"就是 novels[-1]。
    novels: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.novels and not self.characters


def _novels_from_turns(turns: list[dict]) -> list[str]:
    """从落库的 ``sources`` 里提取真实检索到的书名，按最近提到的顺序去重。

    不猜测、不做文本模糊匹配：``sources`` 就是这一轮实际用来生成回答的证据，
    它的 ``novel`` 字段是本项目里能确定"这一轮在聊哪本书"的最强信号——比重新
    在问题文本里做书名匹配更可靠，也不需要额外查库。
    """
    order: dict[str, None] = {}
    for turn in turns:
        for source in turn.get("sources") or []:
            novel = source.get("novel")
            if novel:
                order.pop(novel, None)  # 重新提到就挪到最后，代表"最近还在聊"
                order[novel] = None
    return list(order)


def _known_character_names(novels: list[str]) -> list[str]:
    """查这几本书里人物关系图已确认存在的人名；图未启用或查询失败时返回空表。

    和 ``generation_mixin._graph_hint`` 用的是同一张表、同一条"被拒绝的边不算数"
    规则（M4 审核结论优先于一切自动判断）——这里只是换成"列出人名"而不是
    "查某个具体关系"。
    """
    if not novels:
        return []
    try:
        with connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT person_a AS name FROM character_relations "
                "WHERE novel = ANY(%s) AND COALESCE(review_status, 'pending') <> 'rejected' "
                "UNION SELECT DISTINCT person_b FROM character_relations "
                "WHERE novel = ANY(%s) AND COALESCE(review_status, 'pending') <> 'rejected'",
                (novels, novels),
            ).fetchall()
        return [row["name"] for row in rows if row.get("name")]
    except Exception:
        # 人物关系图可能没建（GRAPH_ENABLED=0）或库暂时不可用：这是纯增强
        # 信息，缺了就退回"只有书名"，不影响正常问答。
        return []


def _characters_mentioned(
    turns: list[dict], known_names: list[str], max_characters: int
) -> list[str]:
    """在对话原文里找出确实出现过的已知人名，按最近提到的顺序去重并截断。

    只在"图里已确认存在的人名"里找——不对自由文本做命名实体识别，避免把
    "他""这个人"之类误判成人名。截断策略与逐字历史一致：牺牲最旧提到的，
    保留最近还在聊的那几个。
    """
    if not known_names:
        return []
    order: dict[str, None] = {}
    for turn in turns:
        text = turn.get("content") or ""
        if not text:
            continue
        for name in known_names:
            if name in text:
                order.pop(name, None)
                order[name] = None
    names = list(order)
    return names[-max_characters:] if max_characters > 0 else names


def extract_session_facts(
    turns: list[dict], max_characters: int = SESSION_FACTS_MAX_CHARACTERS
) -> SessionFacts:
    """从会话历史提取结构化事实。空历史或没有可用证据时返回空结果。"""
    novels = _novels_from_turns(turns)
    known_names = _known_character_names(novels)
    characters = _characters_mentioned(turns, known_names, max_characters)
    return SessionFacts(novels=novels, characters=characters)


def format_facts_line(facts: SessionFacts) -> str:
    """把结构化事实压成一行文本，供拼进「对话背景」。"""
    if facts.is_empty():
        return ""
    parts = []
    if facts.novels:
        current = _display_title(facts.novels[-1])
        others = [_display_title(n) for n in facts.novels[:-1]]
        if others:
            parts.append(f"当前小说：{current}（此前也聊过：{'、'.join(others)}）")
        else:
            parts.append(f"当前小说：{current}")
    if facts.characters:
        parts.append("提到过的人物：" + "、".join(facts.characters))
    return "；".join(parts)
