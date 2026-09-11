from backend.query_cache import CacheKey, QueryCache


def _key(question: str) -> CacheKey:
    return CacheKey(question, "index-v1", "retrieval-v1")


def test_query_cache_is_lru_bounded_and_copies_values():
    cache = QueryCache(max_entries=1)
    original = [{"chunk_id": 1}]
    cache.put(_key("one"), original)
    original[0]["chunk_id"] = 99

    assert cache.get(_key("one")) == [{"chunk_id": 1}]

    cache.put(_key("two"), [{"chunk_id": 2}])
    assert cache.get(_key("one")) is None
    assert cache.snapshot()["evictions"] == 1


def test_query_cache_hit_rate_and_disabled_mode():
    cache = QueryCache()
    cache.put(_key("one"), [{"chunk_id": 1}])
    assert cache.get(_key("one")) == [{"chunk_id": 1}]
    assert cache.get(_key("missing")) is None
    assert cache.snapshot()["hit_rate"] == 0.5

    disabled = QueryCache(enabled=False)
    disabled.put(_key("one"), [{"chunk_id": 1}])
    assert disabled.get(_key("one")) is None
    assert disabled.snapshot()["hits"] == 0
