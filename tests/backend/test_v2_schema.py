import pytest

from v2_schema import apply_v2_schema, render_v2_schema_sql, v2_schema_statements


class _FakeConnection:
    def __init__(self):
        self.statements = []

    def execute(self, query):
        self.statements.append(query)


def test_v2_ddl_is_idempotent_and_does_not_touch_v1_tables():
    statements = v2_schema_statements(384)
    rendered = render_v2_schema_sql(384)

    assert len(statements) >= 10
    assert all("IF NOT EXISTS" in statement for statement in statements)
    assert "knowledge_v2.collections" in rendered
    assert "knowledge_v2.document_chunks" in rendered
    assert "knowledge_v2.chunk_terms" in rendered
    assert "knowledge_v2.index_manifests" in rendered
    assert "DROP TABLE" not in rendered.upper()
    assert "novel_chunks" not in rendered
    assert "chunk_terms" in rendered


def test_apply_v2_schema_is_explicit_and_mockable():
    conn = _FakeConnection()

    count = apply_v2_schema(conn, 3)

    assert count == len(conn.statements)
    assert conn.statements[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert "vector(3)" in "\n".join(conn.statements)


def test_v2_schema_rejects_invalid_embedding_dimension():
    with pytest.raises(ValueError):
        v2_schema_statements(0)
    with pytest.raises(ValueError):
        v2_schema_statements(True)
