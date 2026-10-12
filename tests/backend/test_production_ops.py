"""Maintenance ordering, backup integrity and retention safety with fake executors."""

import io
import json
import logging
import os
import sqlite3
import tarfile
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from backend.logging_config import JSONFormatter
from backend.production_store import ProductionStore
from backend.retention import prune_documents, prune_history
from scripts.production_init import generate
from scripts.production_ops import Operations, digest, verify_backup


def test_generated_deployment_has_separate_workers_databases_sources(tmp_path):
    root = tmp_path / "private"
    compose = json.loads(generate(root, ["alice", "bob"]).read_text())
    services = compose["services"]
    assert services["worker-alice"]["volumes"] != services["worker-bob"]["volumes"]
    assert services["db-alice"]["volumes"] != services["db-bob"]["volumes"]
    assert services["worker-alice"]["networks"] == ["tenant-alice"]
    assert services["worker-bob"]["networks"] == ["tenant-bob"]
    assert "ports" not in services["worker-alice"]
    assert "ports" not in services["db-bob"]
    assert "prometheus" in services and "alertmanager" in services
    registry = json.loads((root / "tenants.json").read_text())
    assert len({entry["token_sha256"] for entry in registry["tenants"]}) == 2
    assert "WORKER_SECRET=" in (root / "alice.env").read_text()
    assert "TENANT_ID=bob" in (root / "bob.env").read_text()
    with pytest.raises(ValueError, match="empty"):
        generate(root, ["alice"])


@pytest.mark.parametrize("ids", [["../escape"], ["alice", "alice"], []])
def test_deployment_rejects_unsafe_users(tmp_path, ids):
    with pytest.raises(ValueError):
        generate(tmp_path / "private", ids)


def make_backup(path, tenant="alice", unsafe=None):
    path.mkdir()
    (path / "database.dump").write_bytes(b"db")
    (path / "usage.json").write_text('[["2026-10-12",3,1,2]]')
    with tarfile.open(path / "sources.tar", "w") as archive:
        for name in (unsafe or "novels/demo.txt", "knowledge/demo.md"):
            member = tarfile.TarInfo(name)
            member.size = 4
            archive.addfile(member, io.BytesIO(b"demo"))
    manifest = {
        "version": 1,
        "tenant": tenant,
        "sha256": {
            name: digest(path / name)
            for name in ("database.dump", "sources.tar", "usage.json")
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_backup_identity_and_checksums_verified(tmp_path):
    backup = tmp_path / "backup"
    make_backup(backup)
    assert verify_backup(backup, "alice")["tenant"] == "alice"
    with pytest.raises(ValueError, match="different"):
        verify_backup(backup, "bob")
    (backup / "database.dump").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_backup(backup, "alice")


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "novels/../../escape",
        "novels\\..\\escape",
        "C:/escape",
        "other/private",
    ],
)
def test_archive_paths_checked_before_restore(tmp_path, name):
    backup = tmp_path / "backup"
    make_backup(backup, unsafe=name)
    with pytest.raises(ValueError):
        verify_backup(backup, "alice")


def test_restore_preserves_old_sources_and_does_not_reset_quota(tmp_path, monkeypatch):
    private = tmp_path / "private"
    generate(private, ["alice", "bob"])
    store = ProductionStore(private / "control" / "quotas.sqlite")
    with store.transaction() as db:
        db.execute("INSERT INTO usage VALUES ('alice','2026-10-12',20,1,2)")
        db.execute("INSERT INTO usage VALUES ('bob','2026-10-12',5,1,2)")
    source = private / "data" / "alice"
    (source / "novels" / "old.txt").write_text("previous")
    backup = tmp_path / "backup"
    make_backup(backup)
    ops = Operations(private, "alice")
    frozen = []
    commands = []
    monkeypatch.setattr(ops, "freeze", lambda: frozen.append(True))
    monkeypatch.setattr(ops, "command", lambda *args: commands.append(args))
    previous = ops.restore(backup)
    assert frozen
    assert (previous / "novels" / "old.txt").read_text() == "previous"
    assert (source / "novels" / "demo.txt").read_text() == "demo"
    assert (private / "data" / "bob").is_dir()
    assert any("--single-transaction" in args for args in commands)
    assert not any("start" in args for args in commands)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT calls FROM usage WHERE tenant='alice'").fetchone()[0] == 20
        assert db.execute("SELECT calls FROM usage WHERE tenant='bob'").fetchone()[0] == 5
    ops.client.close()


def test_invalid_restore_does_not_stop_or_modify_tenant(tmp_path, monkeypatch):
    private = tmp_path / "private"
    generate(private, ["alice"])
    backup = tmp_path / "backup"
    make_backup(backup, tenant="bob")
    ops = Operations(private, "alice")
    stop = MagicMock()
    monkeypatch.setattr(ops, "freeze", stop)
    with pytest.raises(ValueError):
        ops.restore(backup)
    stop.assert_not_called()
    ops.client.close()


class FakeConnection:
    def __init__(self, has_tables=True):
        self.queries = []
        self.has_tables = has_tables

    def transaction(self):
        return nullcontext()

    def execute(self, query, params=None):
        text = query.as_string() if hasattr(query, "as_string") else query
        self.queries.append((text, params))
        result = MagicMock()
        result.fetchone.return_value = {"name": "exists" if self.has_tables else None, "n": 3}
        result.rowcount = 3
        return result


def test_retention_dry_run_never_deletes():
    conn = FakeConnection()
    result = prune_history(conn, days=30)
    assert result["dry_run"]
    assert all(not query.startswith("DELETE") for query, _ in conn.queries)
    assert any(
        "status IN ('completed','failed','cancelled')" in query for query, _ in conn.queries
    )


def test_retention_apply_removes_derived_summaries_and_all_history_tables():
    conn = FakeConnection()
    result = prune_history(conn, days=30, dry_run=False)
    assert result["rows"]["chat_turns"] == 3
    deletions = [query for query, _ in conn.queries if query.startswith("DELETE")]
    assert any("chat_session_summaries WHERE session_id IN" in query for query in deletions)
    assert any('DELETE FROM "run_events"' in query for query in deletions)
    assert any('DELETE FROM "citation_feedback"' in query for query in deletions)


def test_document_retention_removes_old_orphan_sources(tmp_path):
    novels = tmp_path / "novels"
    knowledge = tmp_path / "knowledge"
    novels.mkdir()
    knowledge.mkdir()
    old = knowledge / "old.md"
    recent = knowledge / "recent.md"
    old.write_text("expired")
    recent.write_text("keep")
    timestamp = (datetime.now(UTC) - timedelta(days=60)).timestamp()
    os.utime(old, (timestamp, timestamp))
    conn = FakeConnection(has_tables=False)
    conn.commit = MagicMock()
    assert (
        prune_documents(conn, days=30, novels_dir=novels, knowledge_dir=knowledge)[
            "source_files"
        ]
        == 1
    )
    assert old.exists()
    prune_documents(conn, days=30, novels_dir=novels, knowledge_dir=knowledge, dry_run=False)
    assert not old.exists()
    assert recent.exists()
    conn.commit.assert_called()


def test_production_logs_omit_question_and_credentials(monkeypatch):
    monkeypatch.setenv("PRODUCTION_WORKER", "1")
    monkeypatch.setenv("TENANT_ID", "alice")
    record = logging.LogRecord(
        "novel_rag.main",
        logging.ERROR,
        "main.py",
        10,
        "question private secret-token",
        (),
        None,
    )
    result = JSONFormatter().format(record)
    assert "private" not in result and "secret-token" not in result
    assert json.loads(result)["tenant_id"] == "alice"


def test_novel_delete_cleans_context_and_extraction_caches(monkeypatch):
    import postgres

    conn = FakeConnection()
    original_execute = conn.execute

    def execute(query, params=None):
        result = original_execute(query, params)
        if isinstance(query, str) and query.startswith("SELECT text"):
            result.fetchall.return_value = [{"text": "private original"}]
        return result

    conn.execute = execute
    monkeypatch.setattr(postgres, "connect", lambda: nullcontext(conn))
    postgres.delete_novel_index("demo")
    deletions = [query for query, _ in conn.queries if query.startswith("DELETE")]
    assert any("chunk_contexts" in query for query in deletions)
    assert any("graph_characters" in query for query in deletions)
    assert any("novel_chunks" in query for query in deletions)


def test_monitoring_rules_have_complete_authenticated_delivery():
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    monitoring = Path(__file__).resolve().parents[2] / "deploy" / "monitoring"
    rules = yaml.safe_load((monitoring / "alerts.yml").read_text())["groups"][0]["rules"]
    assert len(rules) >= 11
    assert len({rule["alert"] for rule in rules}) == len(rules)
    receiver = yaml.safe_load((monitoring / "alertmanager.yml").read_text())["receivers"][0][
        "webhook_configs"
    ][0]
    assert receiver["url"] == "http://gateway:8000/admin/alerts"
    assert receiver["http_config"]["authorization"]["credentials_file"]


@pytest.mark.parametrize(
    "path", ["/workspace/novel-rag/backend/main.py", r"C:\workspace\novel-rag\backend\main.py"]
)
def test_pyright_gate_uses_portable_relative_paths(path):
    from scripts.pyright_check import fingerprint

    assert fingerprint({"file": path, "message": "error"})["file"] == "backend/main.py"
