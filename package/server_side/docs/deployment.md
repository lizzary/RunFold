# RunFold Server 部署与维护

RunFold Server 首期只能部署为一个进程、一个 Uvicorn worker、一个服务副本。启动入口会拒绝
`--workers`、`UVICORN_WORKERS` 或 `WEB_CONCURRENCY` 中大于 1 的值。不要让两个 serve 进程共享同一个
`RUNFOLD_DATA_DIR`。

生产环境应使用专用的低权限操作系统账号运行服务。该账号只需读写 `RUNFOLD_DATA_DIR`；数据目录、
SQLite 文件、`objects`、`lance` 和 `staging` 不应允许其他普通用户读取。API 必须位于提供 HTTPS 的
反向代理之后，并只配置精确的 `RUNFOLD_ALLOWED_ORIGINS`。

`RUNFOLD_OPENAI_BASE_URL` 指向的 embedding 服务会收到经授权发送的文档抽取正文和搜索文本。部署方
必须把它视为受信任的数据处理方，并自行确认其地域、留存、训练使用和访问控制政策。API key 只通过
进程环境提供，不应写入数据目录或备份。

备份必须在服务停机后进行，并把整个 `RUNFOLD_DATA_DIR` 作为同一个一致性单元复制，包括
`runfold.sqlite3`（以及仍存在的 WAL/SHM 文件）、`objects` 和 `lance`。仅备份其中一个存储不能得到
可恢复的一致快照。恢复时也应在停服状态下整体还原，然后先启动一次服务，让本地 reconciliation
完成；该过程不会调用 embedding 服务。

索引配置（embedding 地址身份、模型、维度或分块参数）改变后，先停止 serve，再运行：

```text
python -m runfold_server rebuild-index --actor <active-system-admin-username>
```

该命令会产生 embedding 费用，并把用量和审计归到指定的、当前 active 且直接属于
`system_admin` 的账号。命令中断后不要直接对外服务；正常启动会把未完成文档安全收敛为 failed，之后
可再次停服重建或逐文档 reindex。

回收 LanceDB 已删除片段占用的磁盘空间时，先停止 serve，再运行：

```text
python -m runfold_server compact-index
```

两个维护命令都按停服操作设计，不与 serve 并发运行，也不实现跨进程锁。
