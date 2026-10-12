"""Generate a private Compose deployment: one worker, database and source volume per user."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(directory: Path, ids: list[str]):
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", value) for value in ids)
    ):
        raise ValueError("Use unique lowercase user ids (1-32 characters)")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(
            "Destination must be empty; existing credentials are never overwritten"
        )
    os.chmod(directory, 0o700)
    variables = {"PRODUCTION_ADMIN_TOKEN": secrets.token_urlsafe(48)}
    services = {}
    tenants = []
    for identity in ids:
        token = secrets.token_urlsafe(48)
        worker_secret = secrets.token_urlsafe(48)
        db_password = secrets.token_hex(32)
        key = identity.upper().replace("-", "_")
        variables[f"{key}_TOKEN"] = token
        variables[f"{key}_WORKER_SECRET"] = worker_secret
        for subdir in ("novels", "knowledge"):
            (directory / "data" / identity / subdir).mkdir(parents=True)
        worker, database = f"worker-{identity}", f"db-{identity}"
        env_file = f"{identity}.env"
        (directory / env_file).write_text(
            f"DATABASE_URL=postgresql://novel:{db_password}@{database}:5432/novel_rag?connect_timeout=5\n"
            f"POSTGRES_USER=novel\nPOSTGRES_PASSWORD={db_password}\nPOSTGRES_DB=novel_rag\n"
            f"PRODUCTION_WORKER=1\nTENANT_ID={identity}\nWORKER_SECRET={worker_secret}\n"
            "NOVELS_DIR=/tenant/novels\nKNOWLEDGE_DIR=/tenant/knowledge\n"
            "LOG_FORMAT=json\nMODEL_MAX_OUTPUT_CHARACTERS=16000\n",
            encoding="utf-8",
        )
        tenants.append(
            {
                "id": identity,
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "upstream": f"http://{worker}:8000",
                "data_dir": f"/tenants/{identity}",
                "worker_secret_env": f"{key}_WORKER_SECRET",
            }
        )
        services[database] = {
            "image": "pgvector/pgvector:pg16",
            "env_file": [env_file],
            "volumes": [f"pg-{identity}:/var/lib/postgresql/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U novel -d novel_rag"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 10,
            },
            "networks": [f"tenant-{identity}"],
        }
        services[worker] = {
            "build": {"context": str(ROOT)},
            "env_file": [env_file],
            "environment": {
                "OLLAMA_HOST": "http://host.docker.internal:11434",
                "HF_HOME": "/cache/hf",
            },
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "volumes": [f"./data/{identity}:/tenant", f"hf-{identity}:/cache/hf"],
            "depends_on": {database: {"condition": "service_healthy"}},
            "restart": "unless-stopped",
            "networks": [f"tenant-{identity}"],
            "command": [
                "uv",
                "run",
                "--no-dev",
                "python",
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--no-access-log",
            ],
        }
    (directory / "control").mkdir()
    environment = {
        "TENANTS_CONFIG": "/private/tenants.json",
        "QUOTA_DB": "/control/quotas.sqlite",
        "PRODUCTION_ADMIN_TOKEN": "${PRODUCTION_ADMIN_TOKEN:?required}",
    }
    for entry in tenants:
        name = entry["worker_secret_env"]
        environment[name] = "${" + name + ":?required}"
    services["gateway"] = {
        "build": {"context": str(ROOT)},
        "environment": environment,
        "ports": ["127.0.0.1:8000:8000"],
        "volumes": [
            "./tenants.json:/private/tenants.json:ro",
            "./data:/tenants:ro",
            "./control:/control",
        ],
        "restart": "unless-stopped",
        "depends_on": [f"worker-{identity}" for identity in ids],
        "networks": [f"tenant-{identity}" for identity in ids] + ["edge"],
        "command": [
            "uv",
            "run",
            "--no-dev",
            "python",
            "-m",
            "uvicorn",
            "backend.production:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--no-access-log",
        ],
        "healthcheck": {
            "test": [
                "CMD",
                "python",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)",
            ],
            "interval": "30s",
            "timeout": "5s",
            "retries": 3,
        },
    }
    monitoring = ROOT / "deploy" / "monitoring"
    (directory / "metrics-token").write_text(
        variables["PRODUCTION_ADMIN_TOKEN"], encoding="utf-8"
    )
    (directory / "alertmanager.yml").write_text(
        (monitoring / "alertmanager.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    services["prometheus"] = {
        "image": "prom/prometheus:v3.5.0",
        "restart": "unless-stopped",
        "user": "0:0",
        "networks": ["edge"],
        "ports": ["127.0.0.1:9090:9090"],
        "volumes": [
            f"{monitoring / 'prometheus.yml'}:/etc/prometheus/prometheus.yml:ro",
            f"{monitoring / 'alerts.yml'}:/etc/prometheus/alerts.yml:ro",
            "./metrics-token:/run/secrets/metrics-token:ro",
            "prometheus-data:/prometheus",
        ],
        "command": [
            "--config.file=/etc/prometheus/prometheus.yml",
            "--storage.tsdb.retention.time=30d",
        ],
    }
    services["alertmanager"] = {
        "image": "prom/alertmanager:v0.28.1",
        "restart": "unless-stopped",
        "user": "0:0",
        "networks": ["edge"],
        "ports": ["127.0.0.1:9093:9093"],
        "volumes": [
            "./alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro",
            "./metrics-token:/run/secrets/metrics-token:ro",
            "alertmanager-data:/alertmanager",
        ],
    }
    compose = {
        "name": "novel-rag-production",
        "services": services,
        "networks": {**{f"tenant-{identity}": {} for identity in ids}, "edge": {}},
        "volumes": {
            **{name: {} for identity in ids for name in (f"pg-{identity}", f"hf-{identity}")},
            "prometheus-data": {},
            "alertmanager-data": {},
        },
    }
    (directory / "compose.json").write_text(
        json.dumps(compose, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "tenants.json").write_text(
        json.dumps({"tenants": tenants}, indent=2) + "\n", encoding="utf-8"
    )
    (directory / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in variables.items()), encoding="utf-8"
    )
    for path in directory.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    return directory / "compose.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", nargs="+", required=True)
    parser.add_argument("--directory", type=Path, default=ROOT / "deploy" / "private")
    args = parser.parse_args()
    print(f"Created private deployment: {generate(args.directory.resolve(), args.users)}")
    print("Credentials are in its .env; protect this directory and do not commit it.")
