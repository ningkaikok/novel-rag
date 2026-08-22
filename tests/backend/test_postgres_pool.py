"""PostgreSQL 连接池的单元测试。

全部 mock 掉 psycopg.connect / ConnectionPool——CI 环境没有真实 Postgres，
这里只验证 init_pool/close_pool/connect 之间的调度逻辑本身是对的：
有池子时 connect() 从池子借，没有池子时退化为逐次新建连接，
init_pool 失败时要关掉半初始化的池子、不留着它在后台重连。

（本文件之所以能直接 `import postgres` 而不用先插入 src/ 到 sys.path，
是因为 conftest.py 里 `import backend.main` 已经在整个 pytest 会话开始时
做过这件事——sys.path 的修改是进程级的，不是按文件隔离的。）
"""

import pytest

import postgres


@pytest.fixture(autouse=True)
def reset_pool_state():
    """每个测试前后都确保 postgres._pool 是 None，测试之间不互相污染。"""
    postgres._pool = None
    yield
    postgres._pool = None


def test_connect_falls_back_to_plain_psycopg_without_pool(monkeypatch):
    calls = {}

    def fake_connect(url, row_factory):
        calls["url"] = url
        calls["row_factory"] = row_factory
        return "fake-plain-connection"

    monkeypatch.setattr(postgres.psycopg, "connect", fake_connect)

    result = postgres.connect()

    assert result == "fake-plain-connection"
    assert calls["url"] == postgres.DATABASE_URL


class _FakePool:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.wait_called_with = None
        self.closed = False

    def wait(self, timeout=None):
        self.wait_called_with = timeout

    def close(self):
        self.closed = True

    def connection(self):
        return "fake-pooled-connection-context-manager"


def test_init_pool_success_makes_connect_use_the_pool(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(postgres, "ConnectionPool", lambda *a, **k: fake_pool)

    postgres.init_pool(min_size=2, max_size=5)

    assert postgres._pool is fake_pool
    assert fake_pool.wait_called_with is not None  # 确实调用了 wait() 等待就绪
    assert postgres.connect() == "fake-pooled-connection-context-manager"


def test_init_pool_failure_closes_pool_and_leaves_pool_none(monkeypatch):
    class FailingPool(_FakePool):
        def wait(self, timeout=None):
            raise RuntimeError("数据库连不上")

    fake_pool = FailingPool()
    monkeypatch.setattr(postgres, "ConnectionPool", lambda *a, **k: fake_pool)

    with pytest.raises(RuntimeError, match="数据库连不上"):
        postgres.init_pool()

    # 失败不能留着一个还在后台重连的池子，也不能让 _pool 指向一个没就绪的池子
    assert fake_pool.closed is True
    assert postgres._pool is None


def test_close_pool_is_safe_to_call_when_no_pool_exists():
    """没建过池子时调用 close_pool()：不该报错（比如脚本/测试环境）。"""
    postgres.close_pool()  # 不抛异常就算通过
    assert postgres._pool is None


def test_close_pool_after_init(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(postgres, "ConnectionPool", lambda *a, **k: fake_pool)
    postgres.init_pool()

    postgres.close_pool()

    assert fake_pool.closed is True
    assert postgres._pool is None
