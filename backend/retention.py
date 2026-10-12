"""Tenant-local retention. Caller must pause/drain gateway requests and indexing first."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from psycopg import sql

TIMED_TABLES = {
    "chat_turns": "created_at",
    "citation_feedback": "created_at",
    "citation_judgments": "created_at",
    "run_events": "created_at",
    "index_task_runs": "created_at",
}


def prune_history(conn, *, days: int, dry_run: bool = True) -> dict:
    if days < 1:
        raise ValueError("Retention must be at least one day")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    counts = {}
    with conn.transaction():
        for table, column in TIMED_TABLES.items():
            if (
                conn.execute("SELECT to_regclass(%s) AS name", (table,)).fetchone()["name"]
                is None
            ):
                continue
            where = sql.SQL("{} < %s").format(sql.Identifier(column))
            if table == "index_task_runs":
                where += sql.SQL(" AND status IN ('completed','failed','cancelled')")
            if (
                table == "chat_turns"
                and not dry_run
                and conn.execute(
                    "SELECT to_regclass('chat_session_summaries') AS name"
                ).fetchone()["name"]
            ):
                # A summary can include expired content even when newer turns remain.
                conn.execute(
                    "DELETE FROM chat_session_summaries WHERE session_id IN "
                    "(SELECT session_id FROM chat_turns WHERE created_at < %s)",
                    (cutoff,),
                )
            if dry_run:
                row = conn.execute(
                    sql.SQL("SELECT count(*) AS n FROM {} WHERE ").format(
                        sql.Identifier(table)
                    )
                    + where,
                    (cutoff,),
                ).fetchone()
                counts[table] = row["n"]
            else:
                cursor = conn.execute(
                    sql.SQL("DELETE FROM {} WHERE ").format(sql.Identifier(table)) + where,
                    (cutoff,),
                )
                counts[table] = cursor.rowcount
        if (
            not dry_run
            and conn.execute(
                "SELECT to_regclass('chat_session_summaries') AS name"
            ).fetchone()["name"]
        ):
            conn.execute("DELETE FROM chat_session_summaries WHERE updated_at < %s", (cutoff,))
    return {"dry_run": dry_run, "cutoff": cutoff.isoformat(), "rows": counts}


def prune_documents(conn, *, days: int, novels_dir, knowledge_dir, dry_run: bool = True):
    """Expire whole documents and sources while their worker is stopped.

    File unlink follows database commit; interrupted cleanup is safe to retry.
    Old orphan files are also expired to prevent later reindex resurrecting data.
    """
    from pathlib import Path

    from postgres import delete_novel_index
    from v2_repository import delete_v2_document

    if days < 1:
        raise ValueError("Document retention must be at least one day")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    cutoff_ts = cutoff.timestamp()
    expired = []
    protected_titles = set()
    has_v2 = conn.execute("SELECT to_regclass('knowledge_v2.documents') AS name").fetchone()[
        "name"
    ]
    if has_v2:
        rows = conn.execute("""SELECT d.id,d.title,d.metadata,coalesce(max(dv.created_at),d.created_at) AS last_version
            FROM knowledge_v2.documents d LEFT JOIN knowledge_v2.document_versions dv ON dv.document_id=d.id
            GROUP BY d.id ORDER BY d.id""").fetchall()
        protected_titles = {row["title"] for row in rows if row["last_version"] >= cutoff}
        expired = [row for row in rows if row["last_version"] < cutoff]
    targets = []
    for root in (novels_dir, knowledge_dir):
        root = Path(root).resolve()
        for path in root.iterdir() if root.exists() else []:
            if path.is_symlink() or path.resolve().parent != root:
                raise ValueError("Unsafe document retention target")
            if (
                path.is_file()
                and path.stat().st_mtime < cutoff_ts
                and path.name not in protected_titles
                and path.stem not in protected_titles
            ):
                targets.append(path)
    if not dry_run:
        with conn.transaction():
            for row in expired:
                delete_v2_document(conn, str(row["id"]))
        conn.commit()
        for path in targets:
            if (
                path.parent == Path(novels_dir).resolve()
                and path.suffix.lower() == ".txt"
                and conn.execute("SELECT to_regclass('novel_chunks') AS name").fetchone()[
                    "name"
                ]
            ):
                delete_novel_index(path.stem)
            path.unlink(missing_ok=True)
    return {"dry_run": dry_run, "documents": len(expired), "source_files": len(targets)}
