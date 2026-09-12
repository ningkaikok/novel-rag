from chunk_model import SourceChunk
from v2_shadow_reader import V2ShadowReader


class _Embedder:
    def encode(self, questions, *, normalize_embeddings, show_progress_bar):
        assert questions == ["问题"]
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return [[0.1, 0.2]]


class _Repository:
    def vector_search(self, embedding, *, top_k, scope):
        assert embedding == [0.1, 0.2]
        assert top_k == 2
        assert scope is None
        return []

    def keyword_search(self, terms, *, top_k, scope):
        assert terms
        return []


def test_shadow_reader_reports_v1_candidates_missing_from_v2():
    observation = V2ShadowReader(_Embedder(), _Repository()).observe(
        "问题",
        [SourceChunk(novel="演示", chunk_id=3, text="证据", distance=0.1)],
        top_k=2,
    )

    assert observation.payload() == {
        "v1_count": 1,
        "v2_count": 0,
        "missing_count": 1,
        "extra_count": 0,
        "elapsed_ms": observation.elapsed_ms,
    }
