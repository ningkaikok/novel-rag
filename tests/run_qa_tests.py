"""跑一遍 qa_test_set.json 里的问题，把回答和引用来源落盘成 JSON 结果文件。

用法（在项目根目录，激活 venv 后）：
    python tests/run_qa_tests.py [--model qwen2.5:7b] [--out tests/results_7b.json]

不直接判定对错——只负责跑测试、收集原始结果，正确性由人工对照 qa_test_set.json
里的 ground_truth_note 来评判，写进 TEST_REPORT.md。
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent


def ask(question: str, top_k: int = 5) -> dict:
    """调用 /api/ask，消费 SSE，返回 {"answer": str, "sources": [...]}。"""
    resp = requests.post(
        f"{API}/api/ask",
        json={"question": question, "top_k": top_k},
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()

    sources = []
    answer_parts = []
    event = None
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data:
                continue
            if event == "sources":
                sources = json.loads(data)
            elif event == "token":
                answer_parts.append(json.loads(data))
    return {"answer": "".join(answer_parts), "sources": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="仅用于在结果里记录当前测的模型名，不会改变后端实际配置")
    parser.add_argument("--out", default=None, help="结果输出路径，默认 tests/results_<model>.json")
    args = parser.parse_args()

    health = requests.get(f"{API}/api/health", timeout=10).json()
    if not health.get("ready"):
        print("后端未就绪（书架为空或索引未建立），请先上传书籍并重建索引。", file=sys.stderr)
        sys.exit(1)

    questions = json.loads((ROOT / "tests" / "qa_test_set.json").read_text(encoding="utf-8"))
    model_label = args.model or "unknown"
    out_path = Path(args.out) if args.out else ROOT / "tests" / f"results_{model_label}.json"

    results = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        t0 = time.time()
        r = ask(q["question"])
        elapsed = time.time() - t0
        print(f"  -> {elapsed:.1f}s, {len(r['answer'])} 字, {len(r['sources'])} 个来源片段")
        results.append(
            {
                **q,
                "answer": r["answer"],
                "sources": [
                    {"novel": s["novel"], "chunk_id": s["chunk_id"], "text": s["text"]}
                    for s in r["sources"]
                ],
                "elapsed_seconds": round(elapsed, 1),
            }
        )

    payload = {
        "model": model_label,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
