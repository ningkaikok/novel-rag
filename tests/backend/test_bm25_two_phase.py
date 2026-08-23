"""BM25 两阶段聚合改写（M3.4 性能优化）的回归测试。

被测对象是 ``retrieval_mixins.keyword_retrieve`` 的两阶段实现：
SQL 内每词按 tf 取前 ``BM25_PER_TERM_LIMIT`` 个候选，Python 端融合打分。

测试策略（全部 mock，不需要真实 PostgreSQL）：
- 用一个内存语料（chunk 列表）同时驱动「假连接」和「旧逻辑参照实现」：
  假连接按两阶段协议应答——第一个查询（CTE，以 WITH 开头）返回模拟窗口函数
  截断后的候选行；第二个查询（普通 SELECT）按 (novel, chunk_id) 取回元数据。
  候选行的截断语义在假连接里复刻 ``ROW_NUMBER() ... rn <= N``，因此断言的
  是「新代码真的在按每词 Top-N 查询」而不是把全量行搬回 Python；
- 参照实现是旧单条 SQL 的纯 Python 复刻：全量聚合 → 排序 → Top-K。

三个用例分别回答：小数据集上与旧逻辑严格等价、常见词场景聚合行数确实被削减、
语义近似 trade-off 的行为被如实文档化（多词低频片段可能被漏掉）。
"""

import math

import pytest

import retrieval_mixins
from retrieval_mixins import RetrievalMixin

K1, B = 1.2, 0.75


def _corpus(specs):
    """specs: [(novel, chunk_id, {词: tf}), ...] → 统一补上 token_count 等字段。"""
    chunks = []
    for novel, chunk_id, terms in specs:
        chunks.append(
            {
                "novel": novel,
                "chunk_id": chunk_id,
                "terms": terms,
                "token_count": 100 + chunk_id,  # 确定性的长度差异
                "chapter_title": f"第{chunk_id}章",
                "text": f"{novel}-正文{chunk_id}",
                "context": "",
            }
        )
    return chunks


def _stats(chunks):
    n = float(len(chunks))
    avgdl = sum(c["token_count"] for c in chunks) / n
    df = {}
    for chunk in chunks:
        for term in chunk["terms"]:
            df[term] = df.get(term, 0) + 1
    return n, avgdl, df


def _bm25_term_score(tf, df, token_count, n, avgdl):
    """旧 SQL 里 SUM() 内的那个表达式，逐项对应。"""
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * token_count / avgdl))


def _reference_keyword_retrieve(chunks, terms, top_k):
    """旧实现的纯 Python 复刻：对全部命中行求和、排序、取前 K。

    与旧单条 SQL 的唯一差别是并列分数时按 (novel, chunk_id) 决胜——旧 SQL 的
    并列顺序本就不确定（无第二排序键），测试数据刻意构造出无并列的场景。
    """
    n, avgdl, _df_all = _stats(chunks)
    scores = {}
    for chunk in chunks:
        total = 0.0
        for term in terms:
            tf = chunk["terms"].get(term, 0)
            if tf:
                total += _bm25_term_score(
                    tf, float(_count_df(chunks, term)), chunk["token_count"], n, avgdl
                )
        if total:
            scores[(chunk["novel"], chunk["chunk_id"])] = total
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return ranked


def _count_df(chunks, term):
    return sum(1 for c in chunks if term in c["terms"])


class _FakeConn:
    """按两阶段协议应答的假连接，并记录每次执行的 SQL 和参数供断言。

    - 第一个查询（含 WITH 的 CTE）：返回模拟「每词按 tf 取 Top-N」后的候选行，
      字段与新 SQL 的 SELECT 列表一致（不含 text/context 宽列）；
    - 第二个查询（普通 SELECT 元数据）：按参数里成对的 (novel, chunk_id) 取回。
    """

    def __init__(self, chunks, per_term_limit):
        self.chunks = chunks
        self.per_term_limit = per_term_limit
        self.calls: list[tuple[str, list]] = []
        # 每次查询实际返回的行，供「行数削减」断言使用
        self.served: list[list[dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        params = list(params or [])
        self.calls.append((sql, params))
        rows = self._candidates(params) if "WITH" in sql else self._metadata(params)
        self.served.append(rows)
        return _Result(rows)

    def _candidates(self, params):
        # 参数布局：[*terms, *scope...]；mock 场景不用 only_novels
        terms = [p for p in params if isinstance(p, str)]
        n, avgdl, df = _stats(self.chunks)
        rows = []
        for term in terms:
            matched = [c for c in self.chunks if term in c["terms"]]
            # 复刻「每词按 tf 降序取前 N（并列时按 novel, chunk_id 决胜）」
            # 的 LATERAL 截断语义
            matched.sort(key=lambda c: (-c["terms"][term], c["novel"], c["chunk_id"]))
            for c in matched[: self.per_term_limit]:
                rows.append(
                    {
                        "term": term,
                        "novel": c["novel"],
                        "chunk_id": c["chunk_id"],
                        "tf": c["terms"][term],
                        "df": float(df[term]),
                        "token_count": c["token_count"],
                        "n": n,
                        "avgdl": avgdl,
                    }
                )
        return rows

    def _metadata(self, params):
        wanted = {(params[i], int(params[i + 1])) for i in range(0, len(params), 2)}
        return [
            {
                "novel": c["novel"],
                "chunk_id": c["chunk_id"],
                "chapter_title": c["chapter_title"],
                "text": c["text"],
                "context": c["context"],
            }
            for c in self.chunks
            if (c["novel"], c["chunk_id"]) in wanted
        ]


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def _run(monkeypatch, chunks, question_terms, top_k, per_term_limit):
    conn = _FakeConn(chunks, per_term_limit)
    monkeypatch.setattr(retrieval_mixins, "connect", lambda: conn)
    monkeypatch.setattr(retrieval_mixins, "query_terms", lambda _q: list(question_terms))
    monkeypatch.setattr(retrieval_mixins, "BM25_PER_TERM_LIMIT", per_term_limit)
    service = object.__new__(RetrievalMixin)
    results = service.keyword_retrieve("无所谓", top_k=top_k)
    return results, conn


def test_small_dataset_matches_full_aggregation(monkeypatch):
    """每个词的 df 都 ≤ 每词候选上限时，两阶段结果必须与旧全量聚合完全一致。"""
    chunks = _corpus(
        [("书甲", i, {"青云": 1 + i % 3, "剑": i}) for i in range(1, 7)]
        + [("书乙", 7, {"青云": 4, "洞府": 2})]
    )
    terms = ["青云", "洞府"]

    results, conn = _run(monkeypatch, chunks, terms, top_k=20, per_term_limit=200)

    reference = _reference_keyword_retrieve(chunks, terms, top_k=20)
    assert [(r.novel, r.chunk_id) for r in results] == [k for k, _ in reference]
    for result, (_key, score) in zip(results, reference, strict=True):
        # distance 存的是负分（与向量检索的“越小越好”统一）
        assert -result.distance == pytest.approx(score, rel=1e-9)
        assert result.text.endswith(str(result.chunk_id))
        assert result.chapter_title is not None
    # 候选阶段不携带 text/context 宽列：只有打分必需的窄字段
    served = conn.served[0]
    assert served and set(served[0]) == {
        "term",
        "novel",
        "chunk_id",
        "tf",
        "df",
        "token_count",
        "n",
        "avgdl",
    }
    # 候选查询确实是 LATERAL 每词 Top-N 的形状
    assert "LATERAL" in conn.calls[0][0]


def test_common_term_aggregation_rows_are_truncated(monkeypatch):
    """常见词场景：候选行数从全部命中削减为 词数 × 每词上限，且只取回 Top-K 正文。"""
    # 「韩立」命中 50 个片段（tf 各异），另有一个稀有词只命中 2 个片段
    specs = [("书甲", i, {"韩立": 1 + i % 5}) for i in range(50)]
    specs.append(("书甲", 100, {"韩立": 2, "御灵阵": 3}))
    specs.append(("书甲", 101, {"御灵阵": 1}))
    chunks = _corpus(specs)

    results, conn = _run(monkeypatch, chunks, ["韩立", "御灵阵"], top_k=5, per_term_limit=3)

    # SQL 里真的带了每词上限常量，且按 tf 排序截断（LATERAL 每词 Top-N）
    candidate_sql, candidate_params = conn.calls[0]
    assert "LATERAL" in candidate_sql
    assert "ORDER BY ct.tf DESC" in candidate_sql
    assert "LIMIT 3" in candidate_sql
    assert candidate_params[:2] == ["韩立", "御灵阵"]

    # 聚合行数被削减：旧逻辑要处理 52 行命中，现在只有 3 + 2 = 5 行候选
    assert len(conn.served[0]) == 5

    # 只为最终 Top-K（5）请求元数据：第二个查询的参数是成对的键
    _meta_sql, meta_params = conn.calls[1]
    assert len(meta_params) == 2 * len(results) <= 10
    # 稀有词「御灵阵」的 IDF 远高于常见词「韩立」：含稀有词的片段仍排最前，
    # 说明两阶段没有破坏 BM25 的权重语义
    assert results[0].chunk_id == 100
    assert results[0].novel == "书甲"


def test_multi_term_low_tf_chunk_documents_the_tradeoff(monkeypatch):
    """诚实记录近似语义：多词合计相关性高、但单词 tf 都进不了各自 Top-N 的
    片段会被漏掉——旧逻辑能把它排到第一，新逻辑返回结果里没有它。

    这正是 docstring 里声明的 trade-off：调大 BM25_PER_TERM_LIMIT 可逼近旧行为。
    """
    specs = (
        [("书甲", i, {"韩立": 50}) for i in range(1, 5)]  # A 词的 Top4，挤掉目标片段
        + [("书乙", i, {"洞府": 50}) for i in range(1, 5)]  # B 词的 Top4
        + [("书丙", 9, {"韩立": 5, "洞府": 5})]  # 双词各 tf=5：合计分最高
    )
    chunks = _corpus(specs)

    # 旧逻辑（全量聚合）：双词片段总分最高，排第一
    reference = _reference_keyword_retrieve(chunks, ["韩立", "洞府"], top_k=3)
    assert reference[0][0] == ("书丙", 9)

    # 新逻辑：它在两个词上都进不了各自前 3，被截断丢弃
    results, _conn = _run(monkeypatch, chunks, ["韩立", "洞府"], top_k=3, per_term_limit=3)
    assert ("书丙", 9) not in [(r.novel, r.chunk_id) for r in results]

    # 参数化逃生门：上限放大到 ≥ df 后恢复与旧逻辑一致
    results_wide, _conn2 = _run(
        monkeypatch, chunks, ["韩立", "洞府"], top_k=3, per_term_limit=200
    )
    assert [(r.novel, r.chunk_id) for r in results_wide][:1] == [("书丙", 9)]
