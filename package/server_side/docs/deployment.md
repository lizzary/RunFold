# RunFold Server 部署与维护

RunFold Server 首期只能部署为一个进程、一个 Uvicorn worker、一个服务副本。启动入口会拒绝
`--workers` 大于 1 的值。不要让两个 serve 进程共享同一个 `data.directory`。

## 配置与启动

服务只读取 `--config` 指定的 UTF-8 YAML 文件，不读取环境变量或 `.env`。以仓库中的
`config.example.yaml` 为模板创建真实配置，并确保只有运行服务的低权限操作系统账号可读。配置文件
包含 provider API key，空用户库首次启动时还会暂时包含管理员初始密码，因此不得提交版本控制、复制到
数据目录或纳入普通日志采集。

```text
python -m runfold_server --config C:\secure\runfold\config.yaml
```

未传 `--config` 时读取当前工作目录的 `config.yaml`。生产部署应始终显式传入绝对配置路径。首次启动
成功后，停止服务，从 YAML 删除整个 `auth.bootstrap_admin` 节点，妥善处理包含旧密码的配置副本，再
使用清理后的配置重新启动。

生产环境应使用专用的低权限操作系统账号运行服务。该账号需要读取 YAML 配置，并只需读写
`data.directory` 指向的数据根目录；SQLite 文件、`objects`、`lance` 和 `staging` 不应允许其他普通
用户读取。API 必须位于提供 HTTPS 的反向代理之后，`cors.allowed_origins` 只能配置精确 origin。

`provider.base_url` 指向的 embedding 服务会收到经授权发送的文档抽取正文和搜索文本。部署方必须把
它视为受信任的数据处理方，并自行确认其地域、留存、训练使用和访问控制政策。

## 备份

备份必须在服务停机后进行，并把整个 `data.directory` 作为同一个一致性单元复制，包括
`runfold.sqlite3`（以及仍存在的 WAL/SHM 文件）、`objects` 和 `lance`。仅备份其中一个存储不能得到
可恢复的一致快照。YAML 配置应通过独立的受限权限秘密备份流程保存，不要与数据快照混放。

恢复时也应在停服状态下整体还原数据目录，然后先启动一次服务，让本地 reconciliation 完成；该过程
不会调用 embedding 服务。

## 索引维护

索引配置（embedding 地址身份、模型、维度或分块参数）改变后，先停止 serve，再运行：

```text
python -m runfold_server rebuild-index --config C:\secure\runfold\config.yaml --actor <active-system-admin-username>
```

该命令会产生 embedding 费用，并把用量和审计归到指定的、当前 active 且直接属于
`system_admin` 的账号。命令中断后不要直接对外服务；正常启动会把未完成文档安全收敛为 failed，之后
可再次停服重建或逐文档 reindex。

回收 LanceDB 已删除片段占用的磁盘空间时，先停止 serve，再运行：

```text
python -m runfold_server compact-index --config C:\secure\runfold\config.yaml
```

两个维护命令都按停服操作设计，不与 serve 并发运行，也不实现跨进程锁。
