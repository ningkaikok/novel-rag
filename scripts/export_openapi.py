#!/usr/bin/env python3
"""导出 FastAPI 的 OpenAPI schema，作为前后端类型契约的唯一事实来源。

为什么需要这个脚本（类型契约方案）
----------------------------------
frontend/src/api.ts 曾经手写接口类型，和 backend/schemas.py 的 Pydantic 模型
是两套平行维护的定义——后端加字段、前端不知道，字段漂移只能靠 tsc 偶然发现。
现在改为单向流水线：

    schemas.py (Pydantic) → app.openapi() → openapi.json
        → openapi-typescript → src/api-generated.ts → api.ts 只做 re-export

用法
----
    uv run python scripts/export_openapi.py          # 写入 frontend/openapi.json

改过任何 Pydantic 模型 / 路由签名之后必须重跑，然后 `cd frontend && npm run gen:api`。
CI 的 drift 检查步会验证「生成物与提交的一致」，忘了重新生成就会标红。

导入 backend.main 是安全的：FastAPI lifespan（连库/下载模型）只在启动事件里执行，
纯导入不触发——CI 的冒烟检查早已验证这一点。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 注意：模块名 main 不能被本脚本的函数遮蔽（踩过：def main() 覆盖后报
# 「'function' object has no attribute 'app'」），所以导出函数叫 export_schema。
import backend.main as backend_main  # noqa: E402

OUT = ROOT / "frontend" / "openapi.json"


def export_schema() -> None:
    schema = backend_main.app.openapi()
    # 裸 dict 返回值的端点在 schema 里是空对象 `{}`，openapi-typescript 无法解析；
    # 规范化成 {"type": "object"}（语义等价：任意 JSON 对象）。
    for path_item in schema.get("paths", {}).values():
        for method in ("get", "post", "put", "delete", "patch"):
            operation = path_item.get(method)
            if not operation:
                continue
            for response in operation.get("responses", {}).values():
                content = response.get("content", {})
                for media in content.values():
                    if media.get("schema") == {}:
                        media["schema"] = {"type": "object"}
    # sort_keys 让 diff 稳定：Pydantic 内部 dict 顺序变化不再污染 git history
    OUT.write_text(
        json.dumps(schema, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = len(schema.get("paths", {}))
    models = len(schema.get("components", {}).get("schemas", {}))
    print(f"已导出 {OUT.relative_to(ROOT)}（{paths} 个路径 / {models} 个模型）")


if __name__ == "__main__":
    export_schema()
