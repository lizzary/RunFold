# ruff: noqa: RUF001, RUF002
r"""RunFold Server 全功能 HTTP API 演示。

先启动服务，再从 ``package/server_side`` 目录运行：

    .venv\Scripts\python.exe examples\api_demo.py --admin-username admin

管理员密码默认通过安全提示输入，也可以通过 ``RUNFOLD_ADMIN_PASSWORD`` 提供。
脚本会真实创建两个临时用户、两个角色和一个文档；结束时删除文档和角色、恢复
配额，并禁用临时用户。由于服务没有删除用户 API，用户记录会保留在数据库中。
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

EXPECTED_BUSINESS_OPERATIONS = {
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"),
    ("PUT", "/api/auth/password"),
    ("GET", "/api/access/capabilities"),
    ("GET", "/api/access/users"),
    ("POST", "/api/access/users"),
    ("GET", "/api/access/users/{user_id}"),
    ("PATCH", "/api/access/users/{user_id}"),
    ("PUT", "/api/access/users/{user_id}/password"),
    ("GET", "/api/access/users/{user_id}/roles"),
    ("PUT", "/api/access/users/{user_id}/roles"),
    ("GET", "/api/access/roles"),
    ("POST", "/api/access/roles"),
    ("GET", "/api/access/roles/{role_id}"),
    ("PATCH", "/api/access/roles/{role_id}"),
    ("DELETE", "/api/access/roles/{role_id}"),
    ("PUT", "/api/access/roles/{role_id}/capabilities"),
    ("GET", "/api/usage/me"),
    ("GET", "/api/usage/users/{user_id}"),
    ("PUT", "/api/usage/users/{user_id}/limits"),
    ("GET", "/api/security/audit"),
    ("POST", "/api/rag/documents"),
    ("GET", "/api/rag/documents"),
    ("GET", "/api/rag/documents/{document_id}"),
    ("PATCH", "/api/rag/documents/{document_id}"),
    ("DELETE", "/api/rag/documents/{document_id}"),
    ("GET", "/api/rag/documents/{document_id}/content"),
    ("PUT", "/api/rag/documents/{document_id}/content"),
    ("GET", "/api/rag/documents/{document_id}/text"),
    ("PUT", "/api/rag/documents/{document_id}/text"),
    ("POST", "/api/rag/documents/{document_id}/reindex"),
    ("GET", "/api/rag/documents/{document_id}/acl"),
    ("PUT", "/api/rag/documents/{document_id}/acl"),
    ("POST", "/api/rag/search"),
    ("POST", "/api/agent/runs"),
}

OWNER_CAPABILITIES = [
    "agent.run",
    "rag.document.upload",
    "rag.document.read",
    "rag.document.update",
    "rag.document.delete",
    "rag.document.acl.manage",
    "rag.search",
    "usage.self.read",
]

READER_CAPABILITIES = [
    "rag.document.read",
    "rag.search",
    "usage.self.read",
]


class DemoError(RuntimeError):
    """Raised when an API response does not match the demo contract."""


@dataclass
class DemoState:
    admin_token: str = ""
    owner_token: str = ""
    reader_token: str = ""
    owner: dict[str, Any] = field(default_factory=dict)
    reader: dict[str, Any] = field(default_factory=dict)
    role_ids: list[str] = field(default_factory=list)
    document_id: str | None = None
    quota_user_ids: set[str] = field(default_factory=set)


class RunFoldDemo:
    def __init__(
        self,
        client: httpx.Client,
        *,
        admin_username: str,
        admin_password: str,
    ) -> None:
        self.client = client
        self.admin_username = admin_username
        self.admin_password = admin_password
        self.state = DemoState()
        self.covered: set[tuple[str, str]] = set()
        self.request_count = 0
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        self.suffix = f"{stamp}_{secrets.token_hex(3)}"
        self.owner_username = f"demo_owner_{self.suffix}"
        self.reader_username = f"demo_reader_{self.suffix}"
        self.owner_password = f"Owner-{secrets.token_urlsafe(18)}"
        self.owner_reset_password = f"Reset-{secrets.token_urlsafe(18)}"
        self.owner_final_password = f"Final-{secrets.token_urlsafe(18)}"
        self.reader_password = f"Reader-{secrets.token_urlsafe(18)}"

    def execute(self) -> None:
        completed = False
        try:
            self.run()
            completed = True
        finally:
            self.cleanup()
        if completed:
            self.assert_complete_coverage()

    def run(self) -> None:
        self.system_endpoints()
        self.authentication_and_access_control()
        self.usage_limits()
        self.document_lifecycle_and_search()
        self.agent_runtime()
        self.audit_events()
        self.logout_reader()

    def system_endpoints(self) -> None:
        self.phase("健康检查、OpenAPI 和文档页面")
        live = self.request("GET", "/health/live", description="服务存活检查")
        self.require(self.payload(live).get("status") == "live", "存活响应不正确")

        ready = self.request("GET", "/health/ready", description="依赖就绪检查")
        self.require(self.payload(ready).get("status") == "ready", "就绪响应不正确")

        schema_response = self.request("GET", "/openapi.json", description="读取 OpenAPI")
        schema = self.payload(schema_response)
        operations = self.openapi_operations(schema)
        missing = EXPECTED_BUSINESS_OPERATIONS - operations
        extra = operations - EXPECTED_BUSINESS_OPERATIONS
        self.require(not missing, f"OpenAPI 缺少端点：{sorted(missing)}")
        self.require(not extra, f"OpenAPI 出现未纳入 Demo 的端点：{sorted(extra)}")
        print(f"      OpenAPI 已确认 {len(operations)} 个业务操作")

        self.request("GET", "/docs", description="Swagger UI")
        self.request("GET", "/docs/oauth2-redirect", description="Swagger OAuth2 回调页")
        self.request("GET", "/redoc", description="ReDoc 页面")

        print("      已确认用户只通过 /api/agent/runs 对接 /root Agent")

    def authentication_and_access_control(self) -> None:
        self.phase("认证、用户、角色和能力")
        login = self.request(
            "POST",
            "/api/auth/login",
            json={"username": self.admin_username, "password": self.admin_password},
            description="管理员登录",
        )
        self.state.admin_token = self.token_from(login)
        admin_me = self.request(
            "GET",
            "/api/auth/me",
            token=self.state.admin_token,
            description="读取当前管理员",
        )
        print(f"      当前管理员：{self.payload(admin_me)['username']}")

        capabilities = self.request(
            "GET",
            "/api/access/capabilities",
            token=self.state.admin_token,
            params={"limit": 100, "offset": 0},
            description="列出系统能力",
        )
        known_codes = {item["code"] for item in self.payload(capabilities)["items"]}
        required_codes = set(OWNER_CAPABILITIES + READER_CAPABILITIES)
        self.require(required_codes <= known_codes, "服务缺少 Demo 所需能力")

        self.request(
            "GET",
            "/api/access/users",
            token=self.state.admin_token,
            params={"limit": 10, "offset": 0},
            description="分页列出用户",
        )
        self.request(
            "GET",
            "/api/access/roles",
            token=self.state.admin_token,
            params={"limit": 10, "offset": 0},
            description="分页列出角色",
        )

        owner_role = self.create_role(
            name=f"demo_owner_{self.suffix}",
            description="RunFold API demo document owner",
        )
        reader_role = self.create_role(
            name=f"demo_reader_{self.suffix}",
            description="RunFold API demo document reader",
        )
        self.replace_role_capabilities(owner_role["id"], OWNER_CAPABILITIES)
        self.replace_role_capabilities(reader_role["id"], READER_CAPABILITIES)

        renamed_role = self.request(
            "PATCH",
            f"/api/access/roles/{owner_role['id']}",
            route="/api/access/roles/{role_id}",
            token=self.state.admin_token,
            json={"description": "Updated by the complete API demo"},
            description="修改角色",
        )
        self.require(
            self.payload(renamed_role)["description"] == "Updated by the complete API demo",
            "角色更新未生效",
        )
        self.request(
            "GET",
            f"/api/access/roles/{owner_role['id']}",
            route="/api/access/roles/{role_id}",
            token=self.state.admin_token,
            description="读取角色详情",
        )

        self.state.owner = self.create_user(
            username=self.owner_username,
            display_name="Demo Owner",
            password=self.owner_password,
        )
        self.state.reader = self.create_user(
            username=self.reader_username,
            display_name="Demo Reader",
            password=self.reader_password,
        )

        owner_id = self.state.owner["id"]
        reader_id = self.state.reader["id"]
        owner_detail = self.request(
            "GET",
            f"/api/access/users/{owner_id}",
            route="/api/access/users/{user_id}",
            token=self.state.admin_token,
            description="读取用户详情",
        )
        self.require(self.payload(owner_detail)["username"] == self.owner_username, "用户不匹配")

        updated_owner = self.request(
            "PATCH",
            f"/api/access/users/{owner_id}",
            route="/api/access/users/{user_id}",
            token=self.state.admin_token,
            json={"display_name": "Demo Document Owner"},
            description="修改用户显示名称",
        )
        self.require(
            self.payload(updated_owner)["display_name"] == "Demo Document Owner",
            "用户更新未生效",
        )

        self.replace_user_roles(owner_id, [owner_role["id"]])
        self.replace_user_roles(reader_id, [reader_role["id"]])
        assigned = self.request(
            "GET",
            f"/api/access/users/{owner_id}/roles",
            route="/api/access/users/{user_id}/roles",
            token=self.state.admin_token,
            description="读取用户角色",
        )
        self.require(
            self.payload(assigned)["role_ids"] == [owner_role["id"]],
            "用户角色分配未生效",
        )

        self.request(
            "PUT",
            f"/api/access/users/{owner_id}/password",
            route="/api/access/users/{user_id}/password",
            token=self.state.admin_token,
            json={"new_password": self.owner_reset_password},
            expected=204,
            description="管理员重置用户密码",
        )
        owner_login = self.request(
            "POST",
            "/api/auth/login",
            json={"username": self.owner_username, "password": self.owner_reset_password},
            description="文档所有者登录",
        )
        first_owner_token = self.token_from(owner_login)
        self.request(
            "PUT",
            "/api/auth/password",
            token=first_owner_token,
            json={
                "current_password": self.owner_reset_password,
                "new_password": self.owner_final_password,
            },
            expected=204,
            description="用户修改自己的密码",
        )
        self.request(
            "GET",
            "/api/auth/me",
            token=first_owner_token,
            expected=401,
            description="确认旧会话已被撤销",
        )
        owner_login = self.request(
            "POST",
            "/api/auth/login",
            json={"username": self.owner_username, "password": self.owner_final_password},
            description="使用新密码重新登录",
        )
        self.state.owner_token = self.token_from(owner_login)
        self.request(
            "GET",
            "/api/auth/me",
            token=self.state.owner_token,
            description="读取当前文档所有者",
        )

        reader_login = self.request(
            "POST",
            "/api/auth/login",
            json={"username": self.reader_username, "password": self.reader_password},
            description="只读用户登录",
        )
        self.state.reader_token = self.token_from(reader_login)

    def usage_limits(self) -> None:
        self.phase("配额与使用量")
        owner_id = self.state.owner["id"]
        limits = {
            "max_documents": 5,
            "max_storage_bytes": 5_000_000,
            "monthly_embedding_tokens": 100_000,
            "monthly_agent_tokens": 100_000,
        }
        replaced = self.request(
            "PUT",
            f"/api/usage/users/{owner_id}/limits",
            route="/api/usage/users/{user_id}/limits",
            token=self.state.admin_token,
            json=limits,
            description="设置用户配额",
        )
        self.state.quota_user_ids.add(owner_id)
        self.require(self.payload(replaced)["documents"]["limit"] == 5, "文档配额未生效")

        self.request(
            "GET",
            "/api/usage/me",
            token=self.state.owner_token,
            description="读取自己的用量",
        )
        self.request(
            "GET",
            f"/api/usage/users/{owner_id}",
            route="/api/usage/users/{user_id}",
            token=self.state.admin_token,
            description="管理员读取用户用量",
        )

    def document_lifecycle_and_search(self) -> None:
        self.phase("RAG 文档生命周期、ACL 和语义搜索")
        original = b"RunFold demo knowledge. The launch code is ORANGE-42.\n"
        uploaded = self.request(
            "POST",
            "/api/rag/documents",
            token=self.state.owner_token,
            data={"title": "RunFold Demo Knowledge"},
            files={"file": ("runfold-demo.txt", original, "text/plain")},
            expected=201,
            description="上传并索引文档",
        )
        document = self.payload(uploaded)
        document_id = document["id"]
        self.state.document_id = document_id
        self.require(document["index_state"] == "ready", "上传后的索引未就绪")

        page = self.request(
            "GET",
            "/api/rag/documents",
            token=self.state.owner_token,
            params={"limit": 50, "offset": 0},
            description="列出可读文档",
        )
        listed_ids = {item["id"] for item in self.payload(page)["items"]}
        self.require(document_id in listed_ids, "新文档未出现在列表中")

        self.request(
            "GET",
            f"/api/rag/documents/{document_id}",
            route="/api/rag/documents/{document_id}",
            token=self.state.owner_token,
            description="读取文档元数据",
        )
        updated = self.request(
            "PATCH",
            f"/api/rag/documents/{document_id}",
            route="/api/rag/documents/{document_id}",
            token=self.state.owner_token,
            json={"title": "RunFold Demo Knowledge Updated"},
            description="修改文档标题",
        )
        self.require(self.payload(updated)["title"].endswith("Updated"), "文档标题未更新")

        downloaded = self.request(
            "GET",
            f"/api/rag/documents/{document_id}/content",
            route="/api/rag/documents/{document_id}/content",
            token=self.state.owner_token,
            description="下载文档原始内容",
        )
        self.require(downloaded.content == original, "下载内容与上传内容不一致")

        extracted = self.request(
            "GET",
            f"/api/rag/documents/{document_id}/text",
            route="/api/rag/documents/{document_id}/text",
            token=self.state.owner_token,
            description="读取提取文本",
        )
        self.require("ORANGE-42" in self.payload(extracted)["text"], "提取文本内容不正确")

        initial_acl_response = self.request(
            "GET",
            f"/api/rag/documents/{document_id}/acl",
            route="/api/rag/documents/{document_id}/acl",
            token=self.state.owner_token,
            description="读取文档 ACL",
        )
        grants = list(self.payload(initial_acl_response)["grants"])
        grants.append(
            {
                "user_id": self.state.reader["id"],
                "role_id": None,
                "access_level": 10,
            }
        )
        replaced_acl = self.request(
            "PUT",
            f"/api/rag/documents/{document_id}/acl",
            route="/api/rag/documents/{document_id}/acl",
            token=self.state.owner_token,
            json={"grants": grants},
            description="授予只读用户读取 ACL",
        )
        self.require(len(self.payload(replaced_acl)["grants"]) == 2, "ACL 替换未生效")

        self.request(
            "GET",
            f"/api/rag/documents/{document_id}",
            route="/api/rag/documents/{document_id}",
            token=self.state.reader_token,
            description="只读用户读取授权文档",
        )

        replacement = b"Replacement source. The deployment region is TAIWAN-WEST.\n"
        content_replaced = self.request(
            "PUT",
            f"/api/rag/documents/{document_id}/content",
            route="/api/rag/documents/{document_id}/content",
            token=self.state.owner_token,
            files={"file": ("runfold-replacement.txt", replacement, "text/plain")},
            description="替换上传文件并重建索引",
        )
        self.require(self.payload(content_replaced)["index_state"] == "ready", "内容替换失败")

        final_text = (
            "RunFold final demo text. The support color is violet and the code is RF-9000."
        )
        text_replaced = self.request(
            "PUT",
            f"/api/rag/documents/{document_id}/text",
            route="/api/rag/documents/{document_id}/text",
            token=self.state.owner_token,
            json={"text": final_text},
            description="直接替换 TXT 文本并重建索引",
        )
        self.require(self.payload(text_replaced)["index_state"] == "ready", "文本替换失败")

        reindexed = self.request(
            "POST",
            f"/api/rag/documents/{document_id}/reindex",
            route="/api/rag/documents/{document_id}/reindex",
            token=self.state.owner_token,
            description="重新建立文档索引",
        )
        self.require(self.payload(reindexed)["index_state"] == "ready", "重新索引失败")

        search = self.request(
            "POST",
            "/api/rag/search",
            token=self.state.reader_token,
            json={
                "query": "What is the RunFold support code?",
                "top_k": 3,
                "document_ids": [document_id],
            },
            description="在授权文档中进行语义搜索",
        )
        items = self.payload(search)["items"]
        self.require(items, "搜索没有返回结果")
        self.require(all(item["document_id"] == document_id for item in items), "搜索越权")
        print(f"      搜索返回 {len(items)} 个文本块，首个距离={items[0]['distance']:.6f}")

        current_usage = self.request(
            "GET",
            "/api/usage/me",
            token=self.state.owner_token,
            description="查看索引操作后的用量",
        )
        used_tokens = self.payload(current_usage)["embedding_tokens"]["current"]
        print(f"      文档所有者本月嵌入 Token 用量：{used_tokens}")

        self.request(
            "DELETE",
            f"/api/rag/documents/{document_id}",
            route="/api/rag/documents/{document_id}",
            token=self.state.owner_token,
            expected=204,
            description="删除文档及其索引",
        )
        self.state.document_id = None

    def agent_runtime(self) -> None:
        self.phase("/root Agent 与动态团队编排")
        response = self.request(
            "POST",
            "/api/agent/runs",
            token=self.state.owner_token,
            json={
                "input": (
                    "Confirm that the RunFold agent endpoint is operational. "
                    "Return a short final answer."
                )
            },
            description="由 /root 处理用户需求",
        )
        payload = self.payload(response)
        self.require(bool(payload["answer"].strip()), "/root 没有返回最终回答")
        self.require(payload["max_depth_reached"] >= 0, "Agent 深度统计无效")
        print(
            "      /root 已返回最终回答，"
            f"创建员工={payload['agents_created']}，最大深度={payload['max_depth_reached']}"
        )

    def audit_events(self) -> None:
        self.phase("安全审计")
        audit = self.request(
            "GET",
            "/api/security/audit",
            token=self.state.admin_token,
            params={
                "actor_user_id": self.state.owner["id"],
                "limit": 100,
                "offset": 0,
            },
            description="查询文档所有者的审计事件",
        )
        payload = self.payload(audit)
        self.require(payload["items"], "没有查到 Demo 产生的审计事件")
        actions = sorted({item["action"] for item in payload["items"]})
        print(f"      已读取 {len(payload['items'])} 条事件，动作示例：{', '.join(actions[:5])}")

    def logout_reader(self) -> None:
        self.phase("注销会话")
        self.request(
            "POST",
            "/api/auth/logout",
            token=self.state.reader_token,
            expected=204,
            description="注销只读用户当前会话",
        )
        self.state.reader_token = ""

    def cleanup(self) -> None:
        if not self.state.admin_token:
            return
        self.phase("清理 Demo 资源")

        if self.state.document_id is not None:
            self.cleanup_request(
                "DELETE",
                f"/api/rag/documents/{self.state.document_id}",
                route="/api/rag/documents/{document_id}",
                expected={204, 404},
                description="删除残留文档",
            )
            self.state.document_id = None

        for user_id in sorted(self.state.quota_user_ids):
            self.cleanup_request(
                "PUT",
                f"/api/usage/users/{user_id}/limits",
                route="/api/usage/users/{user_id}/limits",
                expected={200, 404},
                json={
                    "max_documents": None,
                    "max_storage_bytes": None,
                    "monthly_embedding_tokens": None,
                    "monthly_agent_tokens": None,
                },
                description="恢复用户默认配额",
            )

        for user in (self.state.owner, self.state.reader):
            if not user:
                continue
            self.cleanup_request(
                "PUT",
                f"/api/access/users/{user['id']}/roles",
                route="/api/access/users/{user_id}/roles",
                expected={200, 404},
                json={"role_ids": []},
                description=f"移除 {user['username']} 的临时角色",
            )

        for role_id in reversed(self.state.role_ids):
            self.cleanup_request(
                "DELETE",
                f"/api/access/roles/{role_id}",
                route="/api/access/roles/{role_id}",
                expected={204, 404},
                description="删除临时角色",
            )

        for user in (self.state.owner, self.state.reader):
            if not user:
                continue
            self.cleanup_request(
                "PATCH",
                f"/api/access/users/{user['id']}",
                route="/api/access/users/{user_id}",
                expected={200, 404},
                json={"status": "disabled"},
                description=f"禁用临时用户 {user['username']}",
            )

    def create_role(self, *, name: str, description: str) -> dict[str, Any]:
        response = self.request(
            "POST",
            "/api/access/roles",
            token=self.state.admin_token,
            json={"name": name, "description": description},
            expected=201,
            description=f"创建角色 {name}",
        )
        role = self.payload(response)
        self.state.role_ids.append(role["id"])
        return role

    def replace_role_capabilities(self, role_id: str, codes: list[str]) -> None:
        response = self.request(
            "PUT",
            f"/api/access/roles/{role_id}/capabilities",
            route="/api/access/roles/{role_id}/capabilities",
            token=self.state.admin_token,
            json={"capability_codes": codes},
            description="替换角色能力",
        )
        self.require(set(self.payload(response)["capability_codes"]) == set(codes), "能力未生效")

    def create_user(
        self, *, username: str, display_name: str, password: str
    ) -> dict[str, Any]:
        response = self.request(
            "POST",
            "/api/access/users",
            token=self.state.admin_token,
            json={
                "username": username,
                "display_name": display_name,
                "password": password,
            },
            expected=201,
            description=f"创建用户 {username}",
        )
        return self.payload(response)

    def replace_user_roles(self, user_id: str, role_ids: list[str]) -> None:
        response = self.request(
            "PUT",
            f"/api/access/users/{user_id}/roles",
            route="/api/access/users/{user_id}/roles",
            token=self.state.admin_token,
            json={"role_ids": role_ids},
            description="替换用户角色",
        )
        self.require(self.payload(response)["role_ids"] == role_ids, "用户角色未生效")

    def cleanup_request(
        self,
        method: str,
        path: str,
        *,
        route: str,
        expected: set[int],
        description: str,
        **kwargs: Any,
    ) -> None:
        try:
            self.request(
                method,
                path,
                route=route,
                token=self.state.admin_token,
                expected=expected,
                description=description,
                **kwargs,
            )
        except (DemoError, httpx.HTTPError) as error:
            print(f"[WARN] 清理失败：{description}：{error}", file=sys.stderr)

    def request(
        self,
        method: str,
        path: str,
        *,
        route: str | None = None,
        token: str | None = None,
        expected: int | set[int] = 200,
        description: str,
        **kwargs: Any,
    ) -> httpx.Response:
        self.request_count += 1
        headers = dict(kwargs.pop("headers", {}))
        headers["X-Request-ID"] = f"demo-{self.request_count}-{secrets.token_hex(4)}"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = self.client.request(method, path, headers=headers, **kwargs)
        allowed = {expected} if isinstance(expected, int) else expected
        marker = "OK" if response.status_code in allowed else "FAIL"
        print(
            f"[{marker}] {method.upper():6} {path:<56} "
            f"{response.status_code:<3} {description}"
        )
        if response.status_code not in allowed:
            body = self.safe_response_text(response)
            raise DemoError(
                f"{method.upper()} {path} 预期状态 {sorted(allowed)}，"
                f"实际为 {response.status_code}：{body}"
            )
        operation = (method.upper(), route or path)
        if operation in EXPECTED_BUSINESS_OPERATIONS:
            self.covered.add(operation)
        return response

    @staticmethod
    def payload(response: httpx.Response) -> dict[str, Any]:
        try:
            value = response.json()
        except ValueError as error:
            raise DemoError(f"响应不是 JSON：{response.text[:500]}") from error
        if not isinstance(value, dict):
            raise DemoError(f"响应 JSON 不是对象：{type(value).__name__}")
        return value

    def token_from(self, response: httpx.Response) -> str:
        token = self.payload(response).get("token")
        self.require(isinstance(token, str) and token, "登录响应缺少 token")
        return token

    @staticmethod
    def openapi_operations(schema: dict[str, Any]) -> set[tuple[str, str]]:
        supported = {"get", "post", "put", "patch", "delete"}
        return {
            (method.upper(), path)
            for path, operations in schema.get("paths", {}).items()
            for method in operations
            if method in supported
        }

    @staticmethod
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise DemoError(message)

    @staticmethod
    def safe_response_text(response: httpx.Response) -> str:
        try:
            value = response.json()
        except ValueError:
            return response.text[:1_000]
        return json.dumps(_redact(value), ensure_ascii=False)[:1_000]

    def assert_complete_coverage(self) -> None:
        missing = EXPECTED_BUSINESS_OPERATIONS - self.covered
        if missing:
            formatted = ", ".join(f"{method} {path}" for method, path in sorted(missing))
            raise DemoError(f"Demo 未成功覆盖以下业务端点：{formatted}")
        print(
            f"\n完成：成功覆盖全部 {len(self.covered)} 个业务 API 操作，"
            f"共发送 {self.request_count} 个 HTTP 请求。"
        )
        print("临时文档和角色已删除，临时用户已禁用，用户记录因无删除 API 而保留。")

    @staticmethod
    def phase(title: str) -> None:
        print(f"\n=== {title} ===")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in {"token", "password", "api_key"} else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="演示 RunFold Server 当前实现的全部 HTTP API 功能。"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RUNFOLD_BASE_URL", "http://127.0.0.1:8000"),
        help="RunFold Server 地址（默认：http://127.0.0.1:8000）",
    )
    parser.add_argument(
        "--admin-username",
        default=os.getenv("RUNFOLD_ADMIN_USERNAME", "admin"),
        help="系统管理员用户名（默认：admin）",
    )
    parser.add_argument(
        "--admin-password",
        default=os.getenv("RUNFOLD_ADMIN_PASSWORD"),
        help="管理员密码；省略时安全提示输入，也可设置 RUNFOLD_ADMIN_PASSWORD",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="单个 HTTP 请求的超时秒数（默认：60）",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="关闭 HTTPS 证书验证，仅限本地测试",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    admin_password = args.admin_password or getpass.getpass("RunFold 管理员密码：")
    if not admin_password:
        print("错误：管理员密码不能为空。", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    print(f"RunFold API Demo -> {base_url}")
    if args.insecure:
        print("警告：HTTPS 证书验证已关闭。", file=sys.stderr)

    try:
        with httpx.Client(
            base_url=base_url,
            timeout=args.timeout,
            follow_redirects=True,
            verify=not args.insecure,
        ) as client:
            demo = RunFoldDemo(
                client,
                admin_username=args.admin_username,
                admin_password=admin_password,
            )
            demo.execute()
    except httpx.RequestError as error:
        print(f"连接失败：{error}", file=sys.stderr)
        return 1
    except DemoError as error:
        print(f"Demo 失败：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDemo 已中断。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
