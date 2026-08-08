"""分词器的单元测试。

分词是 BM25 的地基：建索引和查询必须切出一致的词，否则两边对不上。
这些用例不需要数据库，纯函数测试。
"""
from tokenizer import query_terms, term_frequencies, tokenize


def test_drops_single_chars():
    """单字一律丢掉——中文单字几乎都是虚词，且对缩小检索范围没帮助。"""
    tokens = tokenize("韩立的书")
    assert all(len(t) >= 2 for t in tokens)


def test_drops_stopwords():
    """疑问词/虚词不该进索引，它们在任何问题里都可能出现。"""
    tokens = tokenize("韩立有哪些伴侣")
    assert "哪些" not in tokens
    assert "韩立" in tokens
    assert "伴侣" in tokens


def test_term_frequencies_counts_repeats():
    """文档侧要保留重复次数——这正是 BM25 的 tf 信号。"""
    freqs = term_frequencies("韩立韩立韩立修炼")
    assert freqs["韩立"] == 3
    assert freqs["修炼"] == 1


def test_query_terms_dedupes():
    """查询侧要去重：用户问"韩立和韩立的师父"，「韩立」不该因为重复而权重翻倍。"""
    terms = query_terms("韩立和韩立的师父")
    assert terms.count("韩立") == 1


def test_query_terms_preserves_order():
    """顺序稳定，便于调试时对照——同一个问题每次切出来的词表应该完全一样。"""
    assert query_terms("韩立的师父是谁") == query_terms("韩立的师父是谁")


def test_empty_input():
    assert tokenize("") == []
    assert query_terms("") == []
    assert term_frequencies("") == {}


def test_index_and_query_use_same_rules():
    """核心不变量：同一段文字，走索引侧和查询侧切出的词必须一致
    （查询侧只是额外去了重）。这两边一旦不一致，BM25 就会静默失效——
    索引里存的是 A 词表、查询时拿 B 词表去匹配，永远匹配不上。
    """
    text = "韩立在七玄门修炼长春功"
    assert set(term_frequencies(text)) == set(query_terms(text))
