"""问答路由回归测试：全部是纯规则，不加载模型或数据库。"""

import pytest

from query_router import AnswerMode, build_free_prompt, choose_answer_route


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("你好", AnswerMode.free),
        ("你能做什么？", AnswerMode.free),
        ("什么是 RAG？", AnswerMode.free),
        ("帮我写一首关于月亮的诗", AnswerMode.free),
        ("今天上海天气怎么样", AnswerMode.free),
        ("《凡人修仙传》的结局是什么？", AnswerMode.grounded),
        ("这本书的主角是谁？", AnswerMode.grounded),
        # 没有显式小说词的人名问题必须保守检索，不能让模型凭记忆回答。
        ("韩立的师父是谁？", AnswerMode.grounded),
    ],
)
def test_auto_route(question, expected):
    assert choose_answer_route(question).route is expected


def test_explicit_modes_override_auto_rules():
    assert (
        choose_answer_route("什么是 RAG？", AnswerMode.grounded).route is AnswerMode.grounded
    )
    assert choose_answer_route("韩立的师父是谁？", AnswerMode.free).route is AnswerMode.free


def test_explicit_no_search_wins_in_auto_mode():
    decision = choose_answer_route("不要搜索，简单介绍一下韩立")
    assert decision.route is AnswerMode.free
    assert "不搜索" in decision.reason


def test_free_prompt_discloses_scope():
    prompt = build_free_prompt("什么是 RAG？")
    assert "不检索用户的小说书架" in prompt
    assert "什么是 RAG？" in prompt
