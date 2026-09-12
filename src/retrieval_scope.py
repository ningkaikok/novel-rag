"""把通用 RetrievalScope 安全投影到 V1 小说范围。

该模块只做纯选择，不拼接 SQL。调用者必须提供已从 V1 数据库读出的小说名；版本
范围还必须提供对应 manifest 的 source_hash。无法证明映射关系时返回空集合，绝不
把未知 scope 当成“全部小说”。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from domain_models import RetrievalScope
from legacy_novel import LegacyNovelAdapter


def select_legacy_novels(
    scope: RetrievalScope | None,
    available_novels: Sequence[str],
    *,
    source_hashes: Mapping[str, str | None] | None = None,
) -> list[str] | None:
    """返回 V1 可安全检索的小说名；``None`` 表示未传 scope。"""

    if scope is None:
        return None
    return LegacyNovelAdapter.novel_for_scope(
        scope,
        available_novels,
        source_hashes=source_hashes,
    )
