"""书名识别与问题意图判断的纯函数集合。

初学者可以把这里看成 RAG 的“问题理解层”：在真正开始检索之前，先从用户的
一句话里弄清楚三件事——

    提到哪本书？   _mentions_novel / _novel_titles / _fuzzy_contains
                   （精确匹配失败后用编辑距离容错，容忍同音错字）
    在问什么结构？ _structural_kind（结局 / 开头这类按位置就能回答的问题）
    是不是目录题？ _is_library_question（问书架一共有几本书，走数据库直查）

这些函数刻意不依赖数据库连接和 embedding 模型（``_find_ending_anchor`` 只接收
调用方传入的 conn），因此可以单独做单元测试，也被 ``agent_lab`` 等模块直接复用。

数据类与候选 trace 记录在 ``chunk_model``；检索/生成/流水线编排见各 Mixin 与
``rag`` 模块头部的说明。
"""
import re


def _novel_titles(novel: str) -> list[str]:
    """从库里的书名（文件名）提取用户可能说出的标题。

    文件名形如"《凡人修仙传》（校对版全本+番外）作者：忘语"，用户只会说"凡人修仙传"。
    """
    inner = re.findall(r"《([^》]+)》", novel)
    titles = [name.strip() for name in inner if name.strip()]
    if not titles:
        # 没有书名号就退化为用文件名主体（截断，避免整串带作者名匹配不上）
        titles = [novel.split("（")[0].split("作者")[0].strip()]
    return [t for t in titles if t]


def _edit_distance(a: str, b: str, limit: int) -> int:
    """两字符串的 Levenshtein 距离；一旦确定超过 limit 就提前返回 limit + 1。

    书名很短（通常 3~6 字），这里用滚动数组的朴素实现足够快，不引入额外依赖。
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,  # 删除
                current[j - 1] + 1,  # 插入
                previous[j - 1] + (ca != cb),  # 替换
            )
        # 整行都超过阈值，后面只会更大，可以提前结束
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _fuzzy_contains(question: str, title: str) -> bool:
    """问题里是否有一个与 title 近似的片段（容忍少量错字）。

    用户常打出同音错字，例如把"诡秘之主"打成"闺蜜之主"（guǐ mì / guī mì）。
    精确子串匹配会失败，进而把问题归到别的书上。这里在问题里滑动一个与书名
    等长的窗口，只要某个窗口与书名的编辑距离在容差内就算提到了这本书。

    容差按标题长度取：3 字标题最多错 1 个字，4 字及以上最多错 2 个字
    （"闺蜜之主"→"诡秘之主"就错了 2 个字）。距离上限约为标题长度的一半，
    不同的书之间差异远大于此，不会互相误判。
    """
    n = len(title)
    if n < 3:
        # 标题太短，模糊匹配极易误判，只接受精确包含
        return title in question
    tolerance = 1 if n == 3 else 2
    # 窗口长度允许有 ±tolerance 的浮动，覆盖多字/漏字的情况
    for width in range(max(3, n - tolerance), n + tolerance + 1):
        for start in range(0, len(question) - width + 1):
            window = question[start : start + width]
            if _edit_distance(window, title, tolerance) <= tolerance:
                return True
    return False


def _mentions_novel(question: str, novel: str) -> bool:
    """判断问题里是否提到了这本书（先精确匹配，失败再容错匹配错字）。"""
    titles = _novel_titles(novel)
    if any(title in question for title in titles):
        return True
    return any(_fuzzy_contains(question, title) for title in titles)


# 正文收尾的结构标记。网上流传的 txt 常是"全本+番外"，文件最末往往是番外或
# 作者后记，而不是正文结局，所以要靠这些标记定位真正的结局位置。
_ENDING_MARKERS = ("全书完", "（大结局）", "(大结局)", "大结局", "全文完", "尾声")


_LIBRARY_QUESTION_RE = re.compile(
    r"(?:一共有|共有|总共有|多少部|几部|多少本|几本|有哪些小说|哪些书|所有小说|全部小说|书架)"
)


def _is_library_question(question: str) -> bool:
    """判断问题是否在问书架的完整目录，而不是小说正文内容。"""
    text = question.strip()
    return bool("小说" in text and _LIBRARY_QUESTION_RE.search(text)) or bool(
        "书架" in text and re.search(r"(?:有|包含|多少|几|哪些|全部)", text)
    )


def _find_ending_anchor(conn, novel: str) -> int | None:
    """找出正文结局所在的片段编号；找不到标记时返回 None（调用方退回文件末尾）。

    取最后一个出现结束标记的片段：既能跳过目录里提前出现的"大结局"字样，
    也能避免把后面的番外/后记误当结局。
    """
    row = conn.execute(
        """
        SELECT MAX(chunk_id) AS anchor
        FROM novel_chunks
        WHERE novel = %s
          AND (text LIKE '%%全书完%%' OR text LIKE '%%大结局%%'
               OR text LIKE '%%全文完%%' OR text LIKE '%%尾声%%')
        """,
        (novel,),
    ).fetchone()
    anchor = row and row["anchor"]
    return int(anchor) if anchor is not None else None


def _strip_novel_titles(question: str, novels: list[str]) -> str:
    """把已经识别出来的书名从问题里去掉，只留下真正要检索的内容。

    **为什么只对 BM25 这一路做，不对语义检索做**：书名的职责是「路由」——
    确定要在哪本书里搜。一旦已经靠它把范围限定到《凡人修仙传》，再拿「凡人」
    「修仙」这两个词去这本书内部做关键词匹配就是纯噪声：整本书都在讲凡人修仙，
    这两个词对区分书内的哪一段毫无价值。

    实测过这个 bug 的代价：问「《凡人修仙传》里，韩立小时候的绰号是什么」时，
    书名切出的「凡人」「修仙」两个词给某个无关片段白送了 14.1 分
    （凡人 7.73 + 修仙 6.40），而真正的关键词「绰号」只贡献 7.24 分——
    结果无关片段以 17.63 : 10.13 压过了正确答案所在的片段。

    语义检索不受这个影响：它编码的是整句话的含义，书名只是让语义更完整的
    上下文，不会像 BM25 那样被拆成独立的词各自累加分数。
    """
    stripped = question
    for novel in novels:
        for title in _novel_titles(novel):
            stripped = stripped.replace(f"《{title}》", " ").replace(title, " ")
    return stripped


def _display_title(novel: str) -> str:
    """把库里的文件名式书名压成用户认得的短标题，如《诡秘之主》。"""
    titles = _novel_titles(novel)
    return f"《{titles[0]}》" if titles else novel


def _named_via_typo(question: str, novel: str) -> bool:
    """这本书是靠错字容错匹配上的（而非精确出现在问题里）——用于思考过程里提示'已纠正错字'。"""
    titles = _novel_titles(novel)
    exact = any(title in question for title in titles)
    return (not exact) and any(_fuzzy_contains(question, t) for t in titles)


def _structural_kind(question: str) -> str | None:
    """判断是不是在问书的结构位置：返回 '结局' / '开头' / None。

    与 positional_retrieve 里的词表保持一致，只是这里对外给出可读的类别名。
    """
    text = question.strip()
    if any(w in text for w in ("结局", "结尾", "最后", "最终", "收尾", "大结局", "结束")):
        return "结局"
    if any(w in text for w in ("开头", "开篇", "最初", "一开始", "起初", "开始时")):
        return "开头"
    return None


def _dominant_novels(sources: list["SourceChunk"]) -> list[str]:
    """从已召回的片段里推断问题主要在问哪本书。

    取命中数最多的书；若有其他书命中数达到它的一半以上，则一并保留
    （问题可能确实跨书，例如"两本书的结局有什么不同"）。
    """
    if not sources:
        return []
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.novel] = counts.get(source.novel, 0) + 1
    top = max(counts.values())
    return [novel for novel, n in counts.items() if n * 2 >= top]
