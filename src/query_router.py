"""问答路径路由：决定先检索小说，还是直接使用模型的通用能力。

这是一个刻意保守的路由器。自动模式只在“高置信度属于开放问题”时跳过检索；
拿不准就走原文问答，避免把人物名等小说问题误送进自由问答而产生幻觉。

路由只决定是否检索，不负责生成答案。用户还可以显式选择 ``grounded`` 或
``free`` 覆盖自动判断，所以规则偶尔不符合预期时不需要和系统猜谜。

从 Agent 学习角度看，它已经是一个最小“决策节点”：观察用户问题，选择下一条
执行路径，并给出可展示的 reason。但它还不是完整 Agent，因为没有工具循环、状态图
或自主规划。先把这种确定性路由理解清楚，再升级到 LangGraph 条件边会更直观。
"""
import re
from dataclasses import dataclass
from enum import Enum


class AnswerMode(str, Enum):
    """用户选择的问答模式；值也是前后端 API 契约的一部分。"""

    auto = "auto"
    grounded = "grounded"
    free = "free"


@dataclass(frozen=True)
class RouteDecision:
    """路由结果及其可展示原因。``route`` 不会是 ``auto``。"""

    route: AnswerMode
    reason: str


# 明确要求不查资料时，优先级高于其他所有自动规则。
_EXPLICIT_FREE = (
    "不要搜索",
    "不用搜索",
    "无需搜索",
    "不要查原文",
    "不基于小说",
    "自由回答",
    "用通用知识",
)

# 这些词说明问题很可能在问用户书架里的内容。书中人物名无法穷举，所以未命中
# 任何规则时仍默认 grounded；这个词表只负责阻止明显的小说问题被开放规则抢走。
_GROUNDED_SIGNALS = (
    "小说",
    "原文",
    "书中",
    "这本书",
    "本书",
    "章节",
    "第几章",
    "主角",
    "人物",
    "剧情",
    "情节",
    "结局",
    "开头",
    "作者",
    "片段",
)

# 高置信度的通用知识信号词，面向本项目的目标用户（开发者）收录了技术栈和
# 日常问答两类词汇。匹配前问题已被 casefold，所以英文词统一小写书写即可。
# 这份词表刻意宁缺毋滥：漏判一个通用问题只是白做一次检索，代价很小；
# 误判一个小说问题进自由模式，模型就会凭记忆幻觉作答，代价大得多。
_GENERAL_SIGNALS = (
    "rag",
    "langgraph",
    "langchain",
    "人工智能",
    "机器学习",
    "向量数据库",
    "postgresql",
    "mysql",
    "python",
    "java",
    "kettle",
    "编程",
    "代码",
    "天气",
    "几点",
    "几号",
)

# 三类高置信度开放问题的匹配器。问候必须 fullmatch 整句，避免把
# "你好，介绍一下韩立"这种以问候开头的小说问题误放走；创作类请求总是
# 以"写/翻译/推荐…"起头，所以只看句首。
_GREETING_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|hello|hi|早上好|下午好|晚上好|谢谢|再见)[！!。,.，？?\s]*$",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(r"^(请)?(帮我)?(写|创作|翻译|润色|改写|推荐|起)(一|个|首|段|下)?")
_ASSISTANT_RE = re.compile(r"(你是谁|你能做什么|怎么使用(你|这个功能)?|使用帮助)")


def choose_answer_route(question: str, mode: AnswerMode = AnswerMode.auto) -> RouteDecision:
    """返回实际回答路径。

    ``grounded`` 和 ``free`` 是用户的明确选择，绝不再猜。``auto`` 使用免费规则，
    无需额外调用一次 LLM；这种可重复的行为也便于放进离线评测集持续回归。
    """
    if mode is AnswerMode.grounded:
        return RouteDecision(AnswerMode.grounded, "你选择了「仅依据原文」")
    if mode is AnswerMode.free:
        return RouteDecision(AnswerMode.free, "你选择了「自由问答」")

    text = question.strip()
    lowered = text.casefold()

    # 自动规则按优先级从高到低排列：显式指令 > 小说信号 > 闲聊/帮助 >
    # 通用知识 > 创作任务 > 默认原文。前面的规则命中即返回，后面的不再看；
    # 把小说信号排在通用词之前，是为了保证"这本书讲什么"不被词表误抢。
    if any(signal in text for signal in _EXPLICIT_FREE):
        return RouteDecision(AnswerMode.free, "问题明确要求不搜索小说")

    has_book_signal = ("《" in text and "》" in text) or any(
        signal in text for signal in _GROUNDED_SIGNALS
    )
    if has_book_signal:
        return RouteDecision(AnswerMode.grounded, "问题包含书名或小说内容线索")

    if _GREETING_RE.fullmatch(text) or _ASSISTANT_RE.search(text):
        return RouteDecision(AnswerMode.free, "识别为闲聊或使用帮助")

    if any(signal in lowered for signal in _GENERAL_SIGNALS):
        return RouteDecision(AnswerMode.free, "识别为通用知识问题")

    if _CREATIVE_RE.match(text):
        return RouteDecision(AnswerMode.free, "识别为创作或通用任务")

    # 人名无法靠有限词表识别，因此自动模式宁可多做一次检索，也不把潜在的书中
    # 问题直接交给模型凭记忆回答。用户可用“自由问答”显式覆盖。
    return RouteDecision(AnswerMode.grounded, "未确认是开放问题，保守地检索小说原文")


def build_free_prompt(question: str) -> str:
    """构造自由问答 prompt，并明确标注这条回答不以用户书架为依据。"""
    return f"""你是“书虫”阅读助手。现在是自由问答模式：不检索用户的小说书架，
可以使用通用知识回答。不要声称答案来自用户上传的小说；如果问题必须查看某本书
的具体内容才能确定，请提醒用户切换到“仅依据原文”模式。

用户问题：{question}

回答："""
