"""统一错误响应 schema 的测试：验证每一种错误路径返回的都是
`{"error": {"code": ..., "message": ...}}` 这个形状，包括框架自己的
请求校验错误和完全没预料到的异常。
"""
import io

from fastapi.testclient import TestClient

import backend.main as main
from backend.errors import ErrorCode


def _error_body(resp):
    body = resp.json()
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    return body["error"]


def test_upload_no_valid_files(client):
    files = [("files", ("not-a-novel.pdf", io.BytesIO(b"whatever"), "application/pdf"))]
    resp = client.post("/api/books", files=files)

    assert resp.status_code == 400
    error = _error_body(resp)
    assert error["code"] == ErrorCode.no_valid_files


def test_delete_nonexistent_book(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "NOVELS_DIR", tmp_path)

    resp = client.delete("/api/books/不存在的书")

    assert resp.status_code == 404
    error = _error_body(resp)
    assert error["code"] == ErrorCode.book_not_found


def test_search_index_not_ready(client, monkeypatch):
    monkeypatch.setattr(main, "has_index", lambda: False)

    resp = client.get("/api/search", params={"q": "随便"})

    assert resp.status_code == 409
    error = _error_body(resp)
    assert error["code"] == ErrorCode.index_not_ready


def test_ask_index_not_ready(client):
    # state 里没有 "rag" key，等价于索引还没建立
    resp = client.post("/api/ask", json={"question": "随便问问"})

    assert resp.status_code == 409
    error = _error_body(resp)
    assert error["code"] == ErrorCode.index_not_ready


def test_session_read_failure(client, monkeypatch):
    def boom(session_id):
        raise RuntimeError("数据库连不上")

    monkeypatch.setattr(main, "load_turns", boom)

    resp = client.get("/api/sessions/some-id")

    assert resp.status_code == 500
    error = _error_body(resp)
    assert error["code"] == ErrorCode.session_read_failed
    assert "数据库连不上" in error["message"]


def test_validation_error_on_malformed_ask_body(client):
    """请求体里 question 字段类型不对：FastAPI 自动抛的 RequestValidationError
    也要走同一个 envelope，不是框架默认的 {"detail": [...]} 形状。
    """
    resp = client.post("/api/ask", json={"question": 12345})  # 应该是字符串

    assert resp.status_code == 422
    error = _error_body(resp)
    assert error["code"] == ErrorCode.validation_error
    assert "details" in error  # 逐字段的详细信息保留着，方便调试


def test_unexpected_exception_returns_generic_500_without_leaking_details(monkeypatch):
    """任何没被专门处理的异常：不能把内部报错信息（比如这里的
    "泄漏的内部路径信息"）原样返回给调用方，而是走兜底 handler 的通用文案。

    Starlette 的 ServerErrorMiddleware 即使调用了已注册的 500 handler、
    正确发送了响应，事后仍然会把原始异常重新抛出——这是特意设计的，
    方便测试客户端按需选择"要不要在测试里也看到这个异常"
    （见 starlette/middleware/errors.py 源码里的注释）。这里用
    raise_server_exceptions=False 关掉这个重新抛出，因为我们要验证的是
    "客户端实际收到的 HTTP 响应"，不是"这次调用有没有抛 Python 异常"。
    """

    def boom():
        raise RuntimeError("泄漏的内部路径信息：/etc/secret")

    monkeypatch.setattr(main, "has_index", boom)

    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.get("/api/search", params={"q": "随便"})

    assert resp.status_code == 500
    error = _error_body(resp)
    assert error["code"] == ErrorCode.internal_error
    assert "泄漏的内部路径信息" not in error["message"]
