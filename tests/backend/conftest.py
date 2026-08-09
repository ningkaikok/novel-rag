"""后端 pytest 的共享 fixture。

关键设计：TestClient(app) 不用 `with` 语句时不会触发 FastAPI 的 lifespan——
验证过（见 PR 里的说明），所以这里刻意不用 `with`，避免测试真的去加载
embedding 模型、连接 PostgreSQL。每个测试按需要手动往 backend.main.state
这个全局 dict 里塞假数据，测试之间用 fixture 的 yield/finally 清理，
不会互相污染。
"""
import pytest
from fastapi.testclient import TestClient

import backend.main as main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前后都清空 state，避免上一个测试塞的假数据漏到下一个测试里。"""
    main.state.clear()
    main.index_tasks.reset_for_tests()
    yield
    main.index_tasks.reset_for_tests()
    main.state.clear()
