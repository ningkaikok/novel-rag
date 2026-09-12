"""Bounded in-process cache for retrieval results.

The cache stores only ``SourceChunk`` results, never prompts or generated answers.
Its key must include the current index fingerprint and retrieval configuration so
an index rebuild or ranking change cannot reuse stale evidence.
"""

from __future__ import annotations

import copy
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    question: str
    index_fingerprint: str
    retrieval_fingerprint: str
    # Phase 1 预留通用 collection/document/version 范围；空字符串保持旧调用兼容。
    scope_fingerprint: str = ""


class QueryCache:
    """Thread-safe LRU cache with per-request and aggregate hit metrics."""

    def __init__(self, *, enabled: bool | None = None, max_entries: int | None = None):
        self.enabled = (
            enabled
            if enabled is not None
            else os.environ.get("QUERY_CACHE_ENABLED", "1") != "0"
        )
        self.max_entries = max(
            1,
            max_entries
            if max_entries is not None
            else int(os.environ.get("QUERY_CACHE_MAX_ENTRIES", 128)),
        )
        self._entries: OrderedDict[CacheKey, list[Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: CacheKey) -> list[Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(value)

    def put(self, key: CacheKey, value: list[Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = copy.deepcopy(value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self, *, request_hit: bool | None = None) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "request_hit": request_hit,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else None,
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "evictions": self._evictions,
            }
