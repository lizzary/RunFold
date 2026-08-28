# RunFold Server

RunFold Server 是一个单进程 FastAPI 后端，提供本地账号与角色权限、文档 ACL、RAG 上传/检索、用量审计，以及只通过 `/api/agent/runs` 暴露的 `/root` Agent。

架构、安全边界、数据一致性和维护流程见 [`docs/development-maintenance.md`](docs/development-maintenance.md)。服务启动后也可访问：

- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`
- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`

## 1. 本地启动

要求 Python 3.12+。在 `package/server_side` 目录执行：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --requirement requirements-dev.txt
```

复制 `config.example.yaml` 为一份不进入版本控制的真实配置，至少确认：

- `data.directory` 是绝对路径且服务账号可读写；
- `provider.base_url` 以 `/v1` 结尾；
- embedding model 与 `provider.embedding_dimensions` 匹配；
- chat/embedding provider 可用；
- `cors.allowed_origins` 是精确 origin；
- 首次空库启动时存在 `auth.bootstrap_admin`，密码至少 12 个字符。

启动一个 worker：

```powershell
.venv\Scripts\python.exe -m runfold_server serve --config D:\secure\runfold\config.yaml --workers 1
```

未传 `--config` 时默认读取当前目录的 `config.yaml`。检查服务：

```text
GET http://127.0.0.1:8000/health/live
GET http://127.0.0.1:8000/health/ready
```

预期分别返回：

```json
{"status":"live"}
```

```json
{"status":"ready"}
```

## 2. 初始管理员“注册”方式

服务没有公开 `/register` 或管理员注册 HTTP 接口。空用户库的第一个管理员只能在首次启动时由 YAML bootstrap：

```yaml
auth:
  session_ttl_seconds: 86400
  bootstrap_admin:
    username: admin
    password: replace-with-at-least-12-characters
```

首次启动会创建该用户，并直接分配受保护的 `system_admin` 角色。成功后应：

1. 停止服务；
2. 从 YAML 删除整个 `auth.bootstrap_admin` 节点；
3. 安全处理包含旧密码的配置副本；
4. 使用清理后的配置重新启动。

后续普通用户只能由具有 `identity.user.manage` 的 system_admin 通过 API 创建。

## 3. Postman 环境

下面是一条完整链路：bootstrap 管理员登录 → 创建普通用户 → 分配 reader 角色 → 管理员上传并授权文档 → 普通用户检索 → 普通用户向 Agent 查询。

先在 Postman Environment 创建：

| 变量 | 初值 |
|---|---|
| `base_url` | `http://127.0.0.1:8000` |
| `admin_username` | YAML 中的管理员用户名，例如 `admin` |
| `admin_password` | 首次配置的管理员密码 |
| `demo_username` | `demo_reader` |
| `demo_password` | `demo-reader-password-123` |

后续请求会得到并保存：`admin_token`、`admin_user_id`、`reader_role_id`、`demo_user_id`、`document_id` 和 `user_token`。

除登录外，所有受保护接口统一使用：

```text
Authorization: Bearer {{token}}
```

不要手工设置 multipart 的 `Content-Type`；由 Postman 根据 form-data 自动生成带 boundary 的 header。

## 4. 端到端 HTTP Demo

### 4.1 管理员登录

```http
POST {{base_url}}/api/auth/login
Content-Type: application/json

{
  "username": "{{admin_username}}",
  "password": "{{admin_password}}"
}
```

成功为 `200 OK`：

```json
{
  "token": "opaque-token-returned-once",
  "token_type": "Bearer",
  "expires_at": "2026-08-29T01:02:03+00:00",
  "user": {
    "id": "user-uuid",
    "username": "admin",
    "display_name": "admin",
    "status": "active",
    "created_at": "2026-08-28T01:02:03+00:00",
    "updated_at": "2026-08-28T01:02:03+00:00"
  }
}
```

在 Postman Tests 保存管理员变量：

```javascript
const body = pm.response.json();
pm.environment.set("admin_token", body.token);
pm.environment.set("admin_user_id", body.user.id);
```

### 4.2 查询默认 reader 角色

reader 已由 `schema.sql` 初始化，包含 `agent.run`、`rag.document.read`、`rag.search` 和 `usage.self.read`。

```http
GET {{base_url}}/api/access/roles?limit=100&offset=0
Authorization: Bearer {{admin_token}}
```

响应是授权分页：

```json
{
  "items": [
    {
      "id": "00000000-0000-4000-8000-000000000004",
      "name": "reader",
      "description": "Read authorized knowledge",
      "is_protected": false,
      "capability_codes": [
        "agent.run",
        "rag.document.read",
        "rag.search",
        "usage.self.read"
      ],
      "created_at": "1970-01-01T00:00:00+00:00",
      "updated_at": "1970-01-01T00:00:00+00:00"
    }
  ],
  "limit": 100,
  "offset": 0,
  "total": 4
}
```

实际响应还包含其他角色。Tests 中按名称保存 ID，不依赖数组顺序：

```javascript
const reader = pm.response.json().items.find(item => item.name === "reader");
pm.expect(reader, "reader role must exist").to.exist;
pm.environment.set("reader_role_id", reader.id);
```

### 4.3 管理员创建普通用户

用户名会标准化为小写，必须是 3–64 个 ASCII 字母、数字、点、下划线或连字符；密码必须是 12–256 个字符且不能含控制字符。

```http
POST {{base_url}}/api/access/users
Authorization: Bearer {{admin_token}}
Content-Type: application/json

{
  "username": "{{demo_username}}",
  "display_name": "Demo Reader",
  "password": "{{demo_password}}"
}
```

成功为 `201 Created`，响应是用户对象，但不包含密码或角色。Tests 保存用户 ID：

```javascript
const body = pm.response.json();
pm.environment.set("demo_user_id", body.id);
```

### 4.4 为普通用户完整替换角色集合

这是 `PUT` 完整替换，不是追加。传空数组会移除全部角色。

```http
PUT {{base_url}}/api/access/users/{{demo_user_id}}/roles
Authorization: Bearer {{admin_token}}
Content-Type: application/json

{
  "role_ids": ["{{reader_role_id}}"]
}
```

成功为 `200 OK`：

```json
{
  "user_id": "{{demo_user_id}}",
  "role_ids": ["{{reader_role_id}}"]
}
```

reader 角色只提供全局操作能力；用户能否读取某一份文档仍由文档 ACL 决定。

### 4.5 管理员上传文档

先准备一个本地 `demo.txt`：

```text
RunFold support code is RF-9000.
The support color is violet.
```

在 Postman 中创建请求：

```http
POST {{base_url}}/api/rag/documents
Authorization: Bearer {{admin_token}}
Content-Type: multipart/form-data; boundary=<Postman 自动生成>
```

Body 选择 `form-data`，字段必须且只能各出现一次：

| Key | 类型 | Value |
|---|---|---|
| `title` | Text | `RunFold Demo Knowledge` |
| `file` | File | 选择本地 `demo.txt` |

上传会在当前请求内完成暂存、文本抽取、分块、embedding 和 Lance 索引。成功为 `201 Created`：

```json
{
  "id": "document-uuid",
  "title": "RunFold Demo Knowledge",
  "created_by_user_id": "{{admin_user_id}}",
  "original_filename": "demo.txt",
  "media_type": "text/plain",
  "byte_size": 62,
  "content_hash": "sha256-hex",
  "extracted_characters": 61,
  "chunk_count": 1,
  "index_state": "ready",
  "index_error": null,
  "created_at": "2026-08-28T01:10:00+00:00",
  "updated_at": "2026-08-28T01:10:01+00:00"
}
```

具体 bytes/characters/hash/time 以实际文件为准。Tests 保存文档 ID：

```javascript
const body = pm.response.json();
pm.expect(body.index_state).to.eql("ready");
pm.environment.set("document_id", body.id);
```

上传者会自动获得该文档的直接 MANAGE ACL。

### 4.6 管理员把文档 READ ACL 授给普通用户

ACL 接口也是 `PUT` 完整替换，不能只提交新增项，否则会删除未提交的既有授权。下面同时保留管理员 MANAGE，并授予普通用户 READ：

```http
PUT {{base_url}}/api/rag/documents/{{document_id}}/acl
Authorization: Bearer {{admin_token}}
Content-Type: application/json

{
  "grants": [
    {
      "user_id": "{{admin_user_id}}",
      "access_level": 30
    },
    {
      "user_id": "{{demo_user_id}}",
      "access_level": 10
    }
  ]
}
```

`access_level` 为 `10=READ`、`20=EDIT`、`30=MANAGE`。每条 grant 必须恰好给出 `user_id` 或 `role_id` 之一；可以省略另一个 nullable 字段。成功响应：

```json
{
  "document_id": "{{document_id}}",
  "grants": [
    {
      "user_id": "{{admin_user_id}}",
      "role_id": null,
      "access_level": 30
    },
    {
      "user_id": "{{demo_user_id}}",
      "role_id": null,
      "access_level": 10
    }
  ]
}
```

### 4.7 普通用户登录

```http
POST {{base_url}}/api/auth/login
Content-Type: application/json

{
  "username": "{{demo_username}}",
  "password": "{{demo_password}}"
}
```

Tests 保存普通用户 token：

```javascript
const body = pm.response.json();
pm.environment.set("user_token", body.token);
```

### 4.8 普通用户直接做一次 RAG 检索

这一步不是 Agent 必需步骤，但可以先单独验证 reader capability、文档 READ ACL、query embedding 和向量索引都正常。

```http
POST {{base_url}}/api/rag/search
Authorization: Bearer {{user_token}}
Content-Type: application/json

{
  "query": "What is the RunFold support code?",
  "top_k": 3,
  "document_ids": ["{{document_id}}"]
}
```

成功为 `200 OK`：

```json
{
  "items": [
    {
      "document_id": "{{document_id}}",
      "title": "RunFold Demo Knowledge",
      "ordinal": 0,
      "content_hash": "sha256-hex",
      "text": "RunFold support code is RF-9000.\nThe support color is violet.",
      "distance": 0.123456
    }
  ]
}
```

注意 `document_ids` 的三种语义不同：

- 省略字段：搜索全部当前可读 ready 文档；
- 非空数组：严格限制在指定授权文档；
- 显式 `[]`：返回 `422 invalid_document_scope`，不会扩大为全范围。

### 4.9 普通用户向 `/root` Agent 查询

```http
POST {{base_url}}/api/agent/runs
Authorization: Bearer {{user_token}}
Content-Type: application/json

{
  "input": "请先调用 search_knowledge，在 document_ids [\"{{document_id}}\"] 中搜索 RunFold support code，top_k 使用 3；只根据检索到的证据回答 support code。"
}
```

`thinking_level` 是可选字段；省略时使用 `agent.default_thinking_level`。若显式提供，非空值必须属于配置的 `agent.thinking_level_options`：

```json
{
  "input": "请检索授权知识并回答 RunFold support code。",
  "thinking_level": "xhigh"
}
```

客户端不能传 model、system prompt、tools、skills、用户 ID、权限、团队规模或递归深度。成功为 `200 OK`，结构如下：

```json
{
  "answer": "RunFold support code is RF-9000.",
  "reasoning_content": null,
  "thinking_level": null,
  "agents_created": 0,
  "max_depth_reached": 0
}
```

内容和团队指标由实际模型决策决定；`reasoning_content` 只有 provider 返回时才是字符串。服务只返回 `/root` 最终输出，不返回员工任务、报告、tool transcript 或 system prompt。

## 5. 可选验证与清理请求

### 查看普通用户自己的聚合用量

```http
GET {{base_url}}/api/usage/me
Authorization: Bearer {{user_token}}
```

响应包含文档数、存储 bytes、本月 embedding tokens 和 Agent tokens 的 `current`、`limit`、`remaining`。

### 下载授权原文件

```http
GET {{base_url}}/api/rag/documents/{{document_id}}/content
Authorization: Bearer {{user_token}}
```

响应是原文件字节流，并带 `Content-Disposition: attachment`，不是 JSON。

### 查看规范化抽取文本

```http
GET {{base_url}}/api/rag/documents/{{document_id}}/text
Authorization: Bearer {{user_token}}
```

### 管理员删除文档

```http
DELETE {{base_url}}/api/rag/documents/{{document_id}}
Authorization: Bearer {{admin_token}}
```

成功为 `204 No Content`。删除先将状态切换为 deleting，再清理 Lance rows、对象目录和 SQLite 文档/ACL。

### 注销 session

```http
POST {{base_url}}/api/auth/logout
Authorization: Bearer {{user_token}}
```

成功为 `204 No Content`，该 token 随即失效。

## 6. 常见错误

所有错误采用统一 JSON：

```json
{
  "code": "stable_machine_code",
  "message": "safe human message",
  "details": {},
  "request_id": "request-id"
}
```

| HTTP 状态 / code | 常见原因 |
|---|---|
| `401 invalid_session` | Bearer 缺失、token 错误/过期/已撤销或用户被禁用 |
| `401 invalid_credentials` | 用户名、密码错误或用户 disabled |
| `403 permission_denied` | 角色缺少操作所需的全局 capability |
| `404 document_not_found` | 文档不存在、状态不可见或有 capability 但 ACL 不足；三者故意不区分 |
| `413 upload_too_large` | 文件超过 `rag.upload_max_bytes` |
| `415 unsupported_document_type` | 不是受支持的 txt/md/pdf/docx |
| `422 invalid_request` | JSON 未知字段、字段边界错误，或 multipart 字段名/数量不精确 |
| `422 invalid_document_scope` | RAG 请求显式提交空 `document_ids` |
| `429 quota_exceeded` | 文档、存储、embedding 或 Agent 月度限额已达上限；`details.quota` 指示类别 |
| `502 embedding_provider_error` | embedding provider 连接/状态错误 |
| `502 invalid_embedding_response` | provider 响应条数、顺序、维度、数值或 usage 不符合合约 |
| `502 agent_provider_error` | Agent chat provider 调用失败 |
| `503 service_not_ready` | 本地 SQLite、目录或 Lance readiness 失败 |

响应 header `X-Request-ID` 可与服务端结构化日志和审计关联。请求也可自行传 1–64 字符的安全 `X-Request-ID`。

## 7. 自动化全接口 Demo

仓库还提供 `examples/api_demo.py`，会真实创建临时用户、角色和文档，覆盖全部业务操作并尽量清理。服务已启动时可运行：

```powershell
.venv\Scripts\python.exe examples\api_demo.py --base-url http://127.0.0.1:8000 --admin-username admin
```

管理员密码默认通过安全提示输入。该脚本会调用真实 embeddings 和 Agent provider，产生用量；服务没有删除用户 API，所以临时用户最终只会被禁用，其记录仍保留。

## 8. 开发验证

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

默认测试使用临时 SQLite/LanceDB 和 fake provider。真实 provider E2E 需要显式设置 `RUNFOLD_REAL_E2E=1`，会外发测试文档并产生费用。
