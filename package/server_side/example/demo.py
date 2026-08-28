# ruff: noqa: RUF001, RUF002
r"""管理员登录、上传文件、查询 Agent 的 RunFold 端到端 HTTP Demo。

先启动 RunFold Server，然后从 ``package/server_side`` 目录运行：

    .venv\Scripts\python.exe example\demo.py --admin-username admin

管理员密码默认通过安全提示输入，也可以设置 ``RUNFOLD_ADMIN_PASSWORD``。
省略 ``--file`` 时会上传脚本内置的 Markdown 示例文档。
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import httpx

DEFAULT_DOCUMENT = b"""# RunFold Demo Knowledge

RunFold support code is RF-9000.
The support color is violet.
"""
DEFAULT_QUERY = "What is the RunFold support code?"


class DemoError(RuntimeError):
    """HTTP 响应不符合 RunFold API 合约。"""


def login(
    client: httpx.Client,
    username: str,
    password: str,
) -> tuple[str, dict[str, Any]]:
    """以管理员身份登录并返回 opaque session token。"""
    print("\n[1/3] 管理员登录")
    response = client.post(
        "/api/auth/login",
        headers=_headers("login"),
        json={"username": username, "password": password},
    )
    payload = _expect_json(response, expected_status=200, operation="管理员登录")

    token = payload.get("token")
    user = payload.get("user")
    if not isinstance(token, str) or not token:
        raise DemoError("登录响应缺少非空 token")
    if not isinstance(user, dict):
        raise DemoError("登录响应缺少 user 对象")

    print(f"      已登录：{user.get('username')} ({user.get('id')})")
    print("      token 仅保存在内存中，后续请求使用 Bearer 认证")
    return token, user


def upload_document(
    client: httpx.Client,
    token: str,
    *,
    file_path: Path | None,
    title: str,
) -> dict[str, Any]:
    """通过 multipart/form-data 上传并同步索引文档。"""
    print("\n[2/3] 管理员上传文件")
    stream, filename, media_type = _open_upload(file_path)
    try:
        response = client.post(
            "/api/rag/documents",
            headers=_headers("upload", token),
            data={"title": title},
            files={"file": (filename, stream, media_type)},
        )
    finally:
        stream.close()

    payload = _expect_json(response, expected_status=201, operation="上传文档")
    document_id = payload.get("id")
    if not isinstance(document_id, str) or not document_id:
        raise DemoError("上传响应缺少文档 id")
    if payload.get("index_state") != "ready":
        raise DemoError(f"文档索引未就绪：{payload.get('index_state')!r}")

    print(f"      文件：{filename}")
    print(f"      文档 ID：{document_id}")
    print(f"      索引状态：ready，chunks={payload.get('chunk_count')}")
    return payload


def query_agent(
    client: httpx.Client,
    token: str,
    *,
    query: str,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    """将用户提供的查询原样发送给 /root Agent。"""
    print("\n[3/3] 管理员向 /root Agent 发送查询")
    body: dict[str, Any] = {"input": query}
    if thinking_level is not None:
        body["thinking_level"] = thinking_level

    response = client.post(
        "/api/agent/runs",
        headers=_headers("agent", token),
        json=body,
    )
    payload = _expect_json(response, expected_status=200, operation="Agent 查询")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise DemoError("Agent 响应缺少非空 answer")

    print("\nAgent HTTP 响应：")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def delete_document(client: httpx.Client, token: str, document_id: str) -> None:
    """按需清理本次 Demo 上传的文档。"""
    response = client.delete(
        f"/api/rag/documents/{document_id}",
        headers=_headers("delete", token),
    )
    if response.status_code != 204:
        raise DemoError(_error_message(response, 204, "删除演示文档"))
    print(f"\n已删除演示文档：{document_id}")


def _open_upload(file_path: Path | None) -> tuple[BinaryIO, str, str]:
    if file_path is None:
        return io.BytesIO(DEFAULT_DOCUMENT), "runfold-demo.md", "text/markdown"
    if not file_path.is_file():
        raise DemoError(f"上传文件不存在或不是普通文件：{file_path}")
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    return file_path.open("rb"), file_path.name, media_type


def _headers(operation: str, token: str | None = None) -> dict[str, str]:
    headers = {"X-Request-ID": f"demo-{operation}-{uuid.uuid4().hex[:12]}"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _expect_json(
    response: httpx.Response,
    *,
    expected_status: int,
    operation: str,
) -> dict[str, Any]:
    print(
        f"      {response.request.method} {response.request.url.path} -> "
        f"{response.status_code} (X-Request-ID: "
        f"{response.headers.get('X-Request-ID', '<missing>')})"
    )
    if response.status_code != expected_status:
        raise DemoError(_error_message(response, expected_status, operation))
    try:
        payload = response.json()
    except ValueError as error:
        raise DemoError(f"{operation}响应不是 JSON：{response.text[:500]}") from error
    if not isinstance(payload, dict):
        raise DemoError(f"{operation}响应 JSON 不是对象")
    return payload


def _error_message(
    response: httpx.Response,
    expected_status: int,
    operation: str,
) -> str:
    try:
        body = json.dumps(response.json(), ensure_ascii=False)
    except ValueError:
        body = response.text
    request_id = response.headers.get("X-Request-ID", "<missing>")
    return (
        f"{operation}失败：预期 HTTP {expected_status}，实际 {response.status_code}；"
        f"X-Request-ID={request_id}；响应={body[:1_000]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="管理员登录、上传文档并调用 /root Agent 的端到端 HTTP Demo。"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RUNFOLD_BASE_URL", "http://127.0.0.1:8000"),
        help="RunFold Server 地址（默认：http://127.0.0.1:8000）",
    )
    parser.add_argument(
        "--admin-username",
        default=os.getenv("RUNFOLD_ADMIN_USERNAME", "admin"),
        help="管理员用户名（默认：admin）",
    )
    parser.add_argument(
        "--admin-password",
        default=os.getenv("RUNFOLD_ADMIN_PASSWORD"),
        help="管理员密码；省略时安全提示输入，也可设置 RUNFOLD_ADMIN_PASSWORD",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="要上传的 txt、md、pdf 或 docx；省略时上传内置 Markdown",
    )
    parser.add_argument("--title", help="文档标题；默认使用文件名或内置标题")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Agent 查询（默认：{DEFAULT_QUERY}）",
    )
    parser.add_argument(
        "--thinking-level",
        help="可选；必须属于服务端 agent.thinking_level_options",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单个 HTTP 请求的超时秒数（默认：120）",
    )
    parser.add_argument(
        "--delete-after-run",
        action="store_true",
        help="Agent 返回后删除本次上传的文档",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="关闭 HTTPS 证书验证，仅限本地测试",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = args.admin_password or getpass.getpass("RunFold 管理员密码：")
    if not password:
        print("错误：管理员密码不能为空。", file=sys.stderr)
        return 2
    if not args.query.strip():
        print("错误：query 不能为空。", file=sys.stderr)
        return 2

    file_path = args.file.resolve() if args.file is not None else None
    title = args.title or (file_path.name if file_path else "RunFold Demo Knowledge")
    base_url = args.base_url.rstrip("/")
    print(f"RunFold 端到端 HTTP Demo -> {base_url}")

    try:
        with httpx.Client(
            base_url=base_url,
            timeout=args.timeout,
            follow_redirects=True,
            verify=not args.insecure,
        ) as client:
            token, _ = login(client, args.admin_username, password)
            document = upload_document(
                client,
                token,
                file_path=file_path,
                title=title,
            )
            document_id = str(document["id"])
            try:
                query_agent(
                    client,
                    token,
                    query=args.query,
                    thinking_level=args.thinking_level,
                )
            finally:
                if args.delete_after_run:
                    delete_document(client, token, document_id)
    except (DemoError, httpx.HTTPError, OSError) as error:
        print(f"\nDemo 失败：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDemo 已中断。", file=sys.stderr)
        return 130

    if args.delete_after_run:
        print("\nDemo 完成。")
    else:
        print(f"\nDemo 完成。文档已保留，可稍后通过文档 API 删除：{document_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
