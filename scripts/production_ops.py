"""Offline-consistent tenant backup/restore/purge; pause/drain before touching data.

Backups contain sensitive source/chat content. They must remain private and expire with
the operator's retention policy. Restore requires a reviewed backup from a trusted operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_backup(backup: Path, tenant_id: str):
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("tenant") != tenant_id
    ):
        raise ValueError("Backup belongs to a different tenant or version")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Invalid backup manifest")
    for name in ("database.dump", "sources.tar", "usage.json"):
        if (backup / name).is_symlink() or digest(backup / name) != hashes.get(name):
            raise ValueError("Backup checksum mismatch")
    # Inspect everything before restoring database or writing any file.
    with tarfile.open(backup / "sources.tar") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
                or "\\" in member.name
                or ":" in member.name
            ):
                raise ValueError("Unsafe source archive")
            if not name.parts or name.parts[0] not in {"novels", "knowledge"}:
                raise ValueError("Unexpected source archive root")
    return manifest


class Operations:
    def __init__(self, directory: Path, tenant: str, gateway: str = "http://127.0.0.1:8000"):
        self.directory = directory.resolve()
        registry = json.loads((self.directory / "tenants.json").read_text(encoding="utf-8"))
        if tenant not in {entry["id"] for entry in registry["tenants"]}:
            raise ValueError("Unknown tenant")
        self.tenant = tenant
        self.source = (self.directory / "data" / tenant).resolve()
        if (
            self.source.parent != (self.directory / "data").resolve()
            or self.source.is_symlink()
        ):
            raise ValueError("Invalid tenant source root")
        variables = dict(
            line.split("=", 1)
            for line in (self.directory / ".env").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )
        self.client = httpx.Client(
            base_url=gateway,
            headers={"Authorization": "Bearer " + variables["PRODUCTION_ADMIN_TOKEN"]},
            timeout=35,
            trust_env=False,
        )
        self.compose = [
            "docker",
            "compose",
            "--env-file",
            str(self.directory / ".env"),
            "-f",
            str(self.directory / "compose.json"),
        ]

    def command(self, *args, output=None):
        # Never print Docker output, which can contain environment configuration/secrets.
        result = subprocess.run(
            [*self.compose, *args],
            stdout=output or subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Production operation failed; inspect service logs privately")
        return result.stdout

    def freeze(self):
        deadline = time.monotonic() + 60
        while True:
            response = self.client.post(f"/admin/tenants/{self.tenant}/pause")
            if response.status_code == 200:
                break
            if response.status_code != 409 or time.monotonic() > deadline:
                raise RuntimeError("Tenant could not drain; remains paused")
            time.sleep(1)
        # Graceful shutdown cancels indexing, waits and rolls back unfinished transactions.
        self.command("stop", "--timeout", "60", "worker-" + self.tenant)
        state = json.loads(
            self.command("ps", "--all", "--format", "json", "worker-" + self.tenant)
        )
        rows = state if isinstance(state, list) else [state]
        if not rows or any(row.get("State") != "exited" for row in rows):
            raise RuntimeError("Worker is not stopped; maintenance aborted")

    def resume(self):
        self.command("start", "worker-" + self.tenant)
        response = self.client.post(f"/admin/tenants/{self.tenant}/resume")
        response.raise_for_status()

    def backup(self, destination: Path):
        destination = destination.resolve()
        if (
            destination == self.source
            or self.source in destination.parents
            or destination.exists()
        ):
            raise ValueError("Backup destination must be new and outside the source volume")
        self.freeze()
        destination.mkdir(parents=True)
        os.chmod(destination, 0o700)
        try:
            with (destination / "database.dump").open("wb") as out:
                self.command(
                    "exec",
                    "-T",
                    "db-" + self.tenant,
                    "pg_dump",
                    "-U",
                    "novel",
                    "-d",
                    "novel_rag",
                    "--format=custom",
                    "--no-owner",
                    "--no-acl",
                    output=out,
                )
            with tarfile.open(destination / "sources.tar", "w") as archive:
                for name in ("novels", "knowledge"):
                    path = self.source / name
                    if path.exists():
                        if path.is_symlink() or any(
                            item.is_symlink() or self.source not in item.resolve().parents
                            for item in path.rglob("*")
                        ):
                            raise ValueError("Source symlinks cannot be backed up")
                        archive.add(path, arcname=name)
            with sqlite3.connect(self.directory / "control" / "quotas.sqlite") as db:
                usage = db.execute(
                    "SELECT day,calls,minute,requests FROM usage WHERE tenant=?",
                    (self.tenant,),
                ).fetchall()
            (destination / "usage.json").write_text(json.dumps(usage), encoding="utf-8")
            manifest = {
                "version": 1,
                "tenant": self.tenant,
                "created_at": time.time(),
                "sha256": {
                    name: digest(destination / name)
                    for name in ("database.dump", "sources.tar", "usage.json")
                },
            }
            (destination / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            for file in destination.iterdir():
                os.chmod(file, 0o600)
            verify_backup(destination, self.tenant)
        except BaseException:
            # No manifest means an incomplete backup; never auto-resume on failure.
            (destination / "manifest.json").unlink(missing_ok=True)
            raise
        with sqlite3.connect(self.directory / "control" / "quotas.sqlite") as db:
            db.execute(
                "INSERT INTO metrics VALUES (?,?,?) ON CONFLICT(tenant,name) DO UPDATE SET value=excluded.value",
                (self.tenant, "backup_timestamp_seconds", time.time()),
            )
        self.resume()

    def retention(self, days: int, apply: bool = False, document_days: int | None = None):
        if days < 1 or (document_days is not None and document_days < 1):
            raise ValueError("Retention must be positive")
        self.freeze()
        command = [
            "run",
            "--rm",
            "--no-deps",
            "worker-" + self.tenant,
            "uv",
            "run",
            "--no-dev",
            "python",
            "-m",
            "backend.retention_cli",
            "--days",
            str(days),
        ]
        if document_days is not None:
            command.extend(["--document-days", str(document_days)])
        if apply:
            command.append("--apply")
        return json.loads(self.command(*command))

    def restore(self, backup: Path):
        verify_backup(backup, self.tenant)
        usage = json.loads((backup / "usage.json").read_text(encoding="utf-8"))
        if not isinstance(usage, list) or any(
            not isinstance(row, list)
            or len(row) != 4
            or not isinstance(row[0], str)
            or any(not isinstance(value, int) or value < 0 for value in row[1:])
            for row in usage
        ):
            raise ValueError("Invalid quota snapshot")
        self.freeze()
        # Stage and validate files before database restore; preserve previous source files.
        staged = self.source.with_name(self.tenant + ".restore-" + str(time.time_ns()))
        staged.mkdir()
        with tarfile.open(backup / "sources.tar") as archive:
            archive.extractall(staged, filter="data")
        dump_inside = "/tmp/tenant-restore.dump"
        self.command(
            "cp", str(backup / "database.dump"), "db-" + self.tenant + ":" + dump_inside
        )
        suffix = str(time.time_ns())
        restored_database = "novel_rag_restore_" + suffix
        previous_database = "novel_rag_before_restore_" + suffix
        self.command(
            "exec",
            "-T",
            "db-" + self.tenant,
            "createdb",
            "-U",
            "novel",
            "--template",
            "template0",
            restored_database,
        )
        try:
            self.command(
                "exec",
                "-T",
                "db-" + self.tenant,
                "pg_restore",
                "-U",
                "novel",
                "-d",
                restored_database,
                "--no-owner",
                "--no-acl",
                "--single-transaction",
                "--exit-on-error",
                dump_inside,
            )
        finally:
            self.command("exec", "-T", "db-" + self.tenant, "rm", "-f", dump_inside)
        self.command(
            "exec",
            "-T",
            "db-" + self.tenant,
            "psql",
            "-U",
            "novel",
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            f"BEGIN; ALTER DATABASE novel_rag RENAME TO {previous_database}; ALTER DATABASE {restored_database} RENAME TO novel_rag; COMMIT;",
        )
        # Swap contents inside the existing bind-mounted root (do not rename its inode).
        previous = self.source.with_name(
            self.tenant + ".before-restore-" + str(time.time_ns())
        )
        previous.mkdir()
        for name in ("novels", "knowledge"):
            if (self.source / name).exists():
                shutil.move(str(self.source / name), str(previous / name))
            if (staged / name).exists():
                shutil.move(str(staged / name), str(self.source / name))
            else:
                (self.source / name).mkdir()
        staged.rmdir()
        (previous / "database.json").write_text(
            json.dumps({"database": previous_database}), encoding="utf-8"
        )
        with sqlite3.connect(self.directory / "control" / "quotas.sqlite") as db:
            for day, calls, minute, requests in usage:
                db.execute(
                    "INSERT INTO usage VALUES (?,?,?,?,?) ON CONFLICT(tenant,day) DO UPDATE SET calls=max(calls,excluded.calls)",
                    (self.tenant, day, calls, minute, requests),
                )
        # Remains paused for verification. Resume is a separate explicit operation.
        return previous

    def purge(self):
        snapshots = [
            path
            for pattern in (self.tenant + ".restore-*", self.tenant + ".before-restore-*")
            for path in self.source.parent.glob(pattern)
        ]
        for path in snapshots:
            if path.is_symlink() or path.resolve().parent != self.source.parent:
                raise ValueError("Unsafe restore snapshot deletion target")
        for name in ("novels", "knowledge"):
            target = (self.source / name).resolve()
            if target.parent != self.source or (self.source / name).is_symlink():
                raise ValueError("Unsafe source deletion target")
        self.freeze()
        registry_path = self.directory / "tenants.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for entry in registry["tenants"]:
            if entry["id"] == self.tenant:
                entry["enabled"] = False
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        # Restart gateway to revoke credentials; other workers keep running.
        self.command("restart", "gateway")
        databases = (
            self.command(
                "exec",
                "-T",
                "db-" + self.tenant,
                "psql",
                "-U",
                "novel",
                "-d",
                "postgres",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                "SELECT datname FROM pg_database WHERE datname='novel_rag' OR datname ~ '^novel_rag_(before_restore|restore)_[0-9]+$'",
                "-At",
            )
            .decode()
            .splitlines()
        )
        import re

        if any(
            not re.fullmatch(r"novel_rag(?:_(?:before_restore|restore)_[0-9]+)?", name)
            for name in databases
        ):
            raise ValueError("Unexpected tenant database")
        for name in databases:
            self.command(
                "exec", "-T", "db-" + self.tenant, "dropdb", "-U", "novel", "--force", name
            )
        self.command(
            "exec",
            "-T",
            "db-" + self.tenant,
            "createdb",
            "-U",
            "novel",
            "--template",
            "template0",
            "novel_rag",
        )
        # All targets are resolved and checked before any recursive deletion.
        for name in ("novels", "knowledge"):
            target = (self.source / name).resolve()
            if target.parent != self.source or target.is_symlink():
                raise ValueError("Unsafe source deletion target")
        for name in ("novels", "knowledge"):
            shutil.rmtree(self.source / name)
            (self.source / name).mkdir()
        for snapshot in snapshots:
            shutil.rmtree(snapshot)
        with sqlite3.connect(self.directory / "control" / "quotas.sqlite") as db:
            for table in ("usage", "leases", "metrics", "alerts"):
                db.execute(f"DELETE FROM {table} WHERE tenant=?", (self.tenant,))
        # Remains disabled and stopped; delete external backups per documented policy.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["backup", "verify", "restore", "resume", "purge", "retention"]
    )
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--directory", type=Path, default=ROOT / "deploy" / "private")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--confirm-tenant", help="Required for destructive restore/purge")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--document-days", type=int)
    parser.add_argument(
        "--apply", action="store_true", help="Apply retention (otherwise dry run)"
    )
    args = parser.parse_args()
    if (
        args.action in {"restore", "purge"} or (args.action == "retention" and args.apply)
    ) and args.confirm_tenant != args.tenant:
        parser.error("Destructive operations require --confirm-tenant matching --tenant")
    if args.action in {"backup", "restore", "verify"} and args.backup is None:
        parser.error("--backup is required")
    if args.action == "verify":
        verify_backup(args.backup, args.tenant)
    else:
        operations = Operations(args.directory, args.tenant)
        try:
            if args.action == "backup":
                operations.backup(args.backup)
            elif args.action == "restore":
                print("Previous sources preserved at:", operations.restore(args.backup))
            elif args.action == "retention":
                print(
                    json.dumps(operations.retention(args.days, args.apply, args.document_days))
                )
            else:
                getattr(operations, args.action)()
        finally:
            operations.client.close()
    print("Operation completed")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, httpx.HTTPError):
        print(
            "Operation failed; tenant remains paused after maintenance errors. Inspect privately.",
            file=sys.stderr,
        )
        sys.exit(1)
