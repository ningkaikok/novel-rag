#!/usr/bin/env python3
"""检索实验矩阵：在完全隔离的临时数据库里，对比不同检索配置的质量与成本。

为什么需要这个脚本（路线图 M3.4-①）
----------------------------------
「换个 embedding 模型 / 调一下 chunk size 到底有没有变好」这类问题，靠在生产
索引上手改是答不了的：一是会污染当前可用的书架，二是没有对照就说不清回退。
本脚本把每个实验配置放进一个独立的临时 PostgreSQL 数据库（novel_rag_eval_*），
从零建索引、跑评测、记录指标和成本，最后汇总成一张矩阵表。正式索引全程只读。

每个配置记录什么
----------------
- 配置指纹：配置字典的 SHA-256，保证"同一配置可复现"
- 索引耗时 / 片段数：建索引的成本侧
- Recall@1/3/5、MRR、路由准确率：质量侧（来自 eval_retrieval 的评测逻辑）
- 平均查询延迟：延迟侧
- novel_chunks 表及向量索引的磁盘占用：存储成本侧

用法
----
    # 跑默认矩阵（CI 小语料上的三档 chunk size 对照）
    python scripts/eval_matrix.py

    # 自定义矩阵并落盘
    python scripts/eval_matrix.py --output docs/experiments/matrix_2026w35.json \
        --config '{"name":"overlap160","env":{"CHUNK_OVERLAP":"160"}}' \
        --config '{"name":"chunk300","env":{"CHUNK_SIZE":"300"}}'

安全边界
--------
- 只允许操作 novel_rag_eval_ 前缀的数据库，且默认跑完即删（--keep-db 可保留调试）
- 语料固定用 tests/ci_corpus/（原创文本），绝不把版权小说带进实验库
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CORPUS_DIR = ROOT / "tests" / "ci_corpus"
TEST_SET = ROOT / "tests" / "ci_eval_test_set.json"
DB_PREFIX = "novel_rag_eval_"

# 默认矩阵：M3.4 第一组实验——chunk 粒度对照。语料只有两篇短篇，
# 结论只对"小体量语料 + 当前 embedding 模型"负责；大部头的结论要用
# 同一脚本换语料再跑，不要外推。
DEFAULT_CONFIGS = [
    {"name": "baseline_chunk500", "env": {}},
    {"name": "chunk300", "env": {"CHUNK_SIZE": "300", "CHUNK_OVERLAP": "50"}},
    {"name": "chunk800", "env": {"CHUNK_SIZE": "800", "CHUNK_OVERLAP": "120"}},
]


def config_fingerprint(config: dict) -> str:
    """同一配置得到同一指纹——这是"实验可复现"的最低要求。"""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _run_db_action(db: str, action: str) -> None:
    """createdb/dropdb 的薄封装。drop 用 --if-exists 容忍库不存在；
    注意 createdb 没有这个选项（PostgreSQL 工具的不对称行为），
    所以两条命令的参数分别构造，且都检查退出码——建库失败绝不能静默。"""
    cmd = [action + "db"]
    if action == "drop":
        cmd.append("--if-exists")
    cmd.append(db)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _psql_scalar(db: str, sql: str) -> int:
    out = subprocess.run(
        ["psql", "-d", db, "-tAc", sql], check=True, capture_output=True, text=True
    )
    return int(out.stdout.strip() or 0)


def _child_env(extra: dict) -> dict:
    """给子进程的环境变量。FULL_TEXT_MAX_CHARS 压到 200 强制走真实检索链路，
    否则小语料会触发全文直通、排名失去意义（见 eval_retrieval.py 的说明）。"""
    env = {**os.environ}
    env.setdefault("FULL_TEXT_MAX_CHARS", "200")
    env.update(extra)
    return env


def _index_corpus(db: str, extra_env: dict) -> tuple[float, int]:
    """在临时库里对语料建索引，返回 (耗时秒, 片段总数)。"""
    code = (
        "import sys; from pathlib import Path;"
        f"sys.path.insert(0, {str(ROOT / 'src')!r});"
        "from embedder import load_embedder;"
        "from ingest import build_index;"
        f"result = build_index(load_embedder(), force=True,"
        f" novels_dir=Path({str(CORPUS_DIR)!r}));"
        "print(result['chunk_count'])"
    )
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env({"DATABASE_URL": _db_url(db), **extra_env}),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(f"建索引失败：\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return round(elapsed, 1), int(proc.stdout.strip().splitlines()[-1])


def _db_url(db: str) -> str:
    base = os.environ.get("DATABASE_URL", "")
    if "/" not in base:
        # 没配 DATABASE_URL 时按 config.py 的本机默认拼一个（用户名@127.0.0.1）
        import getpass

        return f"postgresql://{getpass.getuser()}@127.0.0.1:5432/{db}"
    return base.rsplit("/", 1)[0] + f"/{db}"


def _run_eval(db: str, extra_env: dict) -> dict:
    report_path = ROOT / ".eval_matrix_tmp.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_retrieval.py"),
            "--test-set",
            str(TEST_SET),
            "--save",
            str(report_path),
        ],
        env=_child_env({"DATABASE_URL": _db_url(db), **extra_env}),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"评测失败：\n{proc.stderr[-2000:]}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.unlink(missing_ok=True)
    return report


def run_one(config: dict, keep_db: bool = False) -> dict:
    """跑单个配置：建临时库 → 建索引 → 评测 → 记录成本 → 删库。"""
    fingerprint = config_fingerprint(config)
    db = f"{DB_PREFIX}{fingerprint}"
    extra_env = dict(config.get("env", {}))

    _run_db_action(db, "drop")  # 上次中断留下的同名库直接清掉，保证从头建
    _run_db_action(db, "create")
    subprocess.run(
        [
            "psql",
            "-d",
            db,
            "-c",
            "CREATE EXTENSION IF NOT EXISTS vector;",
        ],
        check=True,
        capture_output=True,
    )

    try:
        index_s, chunk_count = _index_corpus(db, extra_env)
    except RuntimeError as exc:
        # 索引失败（最常见：M3.3 质量门禁拦截超长输入）本身就是实验结论——
        # 记为无效配置而不是让整个矩阵中断。
        if not keep_db:
            _run_db_action(db, "drop")
        return {
            **config,
            "fingerprint": fingerprint,
            "database": db,
            "index_seconds": None,
            "chunk_count": None,
            "storage_bytes": None,
            "metrics": {},
            "case_count": 0,
            # 只留最后一段：错误根因（如质量门禁的具体报错）都在 traceback 尾部
            "error": str(exc)[-300:],
        }
    report = _run_eval(db, extra_env)
    storage = _psql_scalar(
        db,
        # 向量索引名以 postgres.py 里的实际 DDL 为准（HNSW）
        "SELECT pg_total_relation_size('novel_chunks')"
        " + COALESCE(pg_total_relation_size('novel_chunks_embedding_hnsw_idx'), 0)",
    )
    if not keep_db:
        _run_db_action(db, "drop")

    summary = report["summary"]
    return {
        **config,
        "fingerprint": fingerprint,
        "database": db,
        "index_seconds": index_s,
        "chunk_count": chunk_count,
        "storage_bytes": storage,
        "metrics": {k: summary[k] for k in summary if k != "cases"},
        "case_count": summary["cases"],
    }


def print_matrix(rows: list[dict]) -> None:
    cols = ["recall@1", "recall@3", "recall@5", "mrr", "routing_accuracy", "avg_ms"]
    head = f"{'配置':<22} {'指纹':<14} {'片段':<6} {'索引s':<7} {'存储KB':<9}" + "".join(
        f"{c:<16}" for c in cols
    )
    print()
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for row in rows:
        if row.get("error"):
            print(f"{row['name']:<22} {row['fingerprint']:<14} ❌ 无效配置：{row['error'][:80]}")
            continue
        line = (
            f"{row['name']:<22} {row['fingerprint']:<14} "
            f"{row['chunk_count']:<6} {row['index_seconds']:<7} "
            f"{row['storage_bytes'] // 1024:<9}"
        )
        line += "".join(f"{row['metrics'].get(c, '-')!s:<16}" for c in cols)
        print(line)
    print()


def main() -> None:
    global CORPUS_DIR, TEST_SET
    parser = argparse.ArgumentParser(description="检索实验矩阵（隔离临时库）")
    parser.add_argument(
        "--config",
        action="append",
        metavar="JSON",
        help="追加一个配置（如 '{\"name\":\"x\",\"env\":{\"CHUNK_SIZE\":\"300\"}}'）；"
        "可重复传多次。不传则使用默认的三档 chunk 对照",
    )
    parser.add_argument(
        "--corpus-dir",
        metavar="DIR",
        help=f"语料目录（默认 {CORPUS_DIR}）。大部头实验指向本地受控样本目录，"
        "注意版权文本不得放进仓库",
    )
    parser.add_argument(
        "--test-set",
        metavar="FILE",
        help=f"评测集（默认 {TEST_SET}），应与语料配套",
    )
    parser.add_argument("--output", metavar="FILE", help="把完整结果写成 JSON 文件")
    parser.add_argument("--keep-db", action="store_true", help="保留临时库供调试（默认跑完即删）")
    args = parser.parse_args()

    if args.corpus_dir:
        CORPUS_DIR = Path(args.corpus_dir).resolve()
        if not CORPUS_DIR.is_dir():
            sys.exit(f"语料目录不存在：{CORPUS_DIR}")
    if args.test_set:
        TEST_SET = Path(args.test_set).resolve()
        if not TEST_SET.is_file():
            sys.exit(f"评测集不存在：{TEST_SET}")

    configs = [json.loads(c) for c in args.config] if args.config else DEFAULT_CONFIGS
    rows = []
    for config in configs:
        print(f"\n▶ 配置 {config['name']}（指纹 {config_fingerprint(config)}）…")
        rows.append(run_one(config, keep_db=args.keep_db))

    print_matrix(rows)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整结果已写入 {out}")


if __name__ == "__main__":
    main()
