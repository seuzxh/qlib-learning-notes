# 服务型示例的环境说明（MongoDB 与 Arctic）

> 配置日期：2026-10-05。为 `examples/{model_rolling, online_srv, orderbook_data}` 三组服务型示例提供运行前提。已核验：这些示例只依赖 MongoDB，**不需要 Redis**。

## 一、MongoDB（Docker 容器，已运行）

- 部署方式：容器 `qlib-mongo`，镜像 `mongo:7`（7.0.43），数据卷 `/home/zxh/mongodb_data`（容器内 `/data/db`），仅绑定 `127.0.0.1:27017`，`--restart unless-stopped`（宿主机重启后自启）。
- 镜像存储在 docker 数据根 `/data01/docker`（独立大盘，不占根分区）。
- qlib 侧接入方式（两处等价）：
  - 代码里 `qlib.init(mongo={"task_url": "mongodb://127.0.0.1:27017", "task_db_name": "你的库名"})`，然后 `TaskManager(task_pool="队列名")`；
  - 示例脚本（`model_rolling/task_manager_rolling.py`、`online_srv/*.py`）里默认写的 `mongodb://10.0.0.4:27017/` 运行时要改成上面的本机地址。
- 常用命令：

```bash
docker ps --filter name=qlib-mongo          # 查看状态
docker logs qlib-mongo --tail 20            # 看日志
docker stop qlib-mongo && docker start qlib-mongo   # 重启
# 快速验证（qlib 环境）
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://127.0.0.1:27017').admin.command('ping'))"
```

## 二、Arctic（独立 conda 环境，已验证）

- 为什么单开环境：旧版 `arctic`（1.82.2）钉 `pymongo 3.x`，与 qlib 环境的 pymongo 4.x 冲突，二者不能共存。
- 环境：`conda activate arctic`（Python 3.8，arctic 1.82.2 + pymongo 3.11.0）。
- 已通过的冒烟：连接本机 MongoDB → `initialize_library(lib, CHUNK_STORE)` → 写入/读回按日分块 DataFrame（sharding 警告属单节点正常现象，可忽略）。
- `orderbook_data/create_dataset.py`（导入原始数据到 Arctic）须在 `arctic` 环境下运行；`example.py`（qlib 侧查询）在 `qlib` 环境下运行。
- 注意脚本内 MongoDB 地址与库库名要指向本机（原脚本按官方环境写死）。

## 三、orderbook 示例数据

- 官方来源：Google Drive（`gdown https://drive.google.com/uc?id=15FuUqWn2rkCi8uhJYGEQWKakcEqLJNDG`），需代理；本机代理 `http://127.0.0.1:7890`。下载状态见文末补记。
- `example.py` 还引用 `~/.qlib/qlib_data/yahoo_cn_1min` 的 1min 数据；本机等价物是 `~/.qlib/qlib_data/cn_data_1min`，运行时需相应适配（provider_uri 与 ArcticFeatureProvider 配置）。

## 四、四组服务型示例的就绪状态

| 示例 | 前提 | 状态 |
|---|---|---|
| model_rolling/ | MongoDB | **就绪**（运行前把脚本日期窗口与 mongo 地址适配即可） |
| online_srv/ | MongoDB | **就绪**（同上，另注意其测试配置钉在 2013~2020 CSI100） |
| orderbook_data/ | MongoDB + arctic 环境 + 示例数据 | 环境**就绪**；数据下载见补记 |
| rl/、rl_order_execution/ | torch + gym + tianshou | **simple_example 已跑通**（关键：tianshou 必须装 pyproject 钉的 0.4.10，2.0/0.5 均不兼容；qlib 环境另装了 shimmy 备用）；rl_order_execution 为数小时级长流程未跑 |

## 补记：数据下载结果

无需下载——目录里**早已有完整的 `orderbook_example_data.zip`（2.50GB，2025-08-19 落盘）**，内容为 2020-12-31 单日的六类订单簿数据（Tick/Order/OrderQueue/Transaction/Minute/Day 的 tar.gz 与股票清单 all.txt），已解压至 `examples/orderbook_data/orig_data/`（2.4GB，已加入 .gitignore 防入库）。重试的 gdown 半成品已清理。

## 五、就绪总览

四组服务型示例的运行前提（MongoDB、arctic 环境、订单簿数据）**已全部就绪**；后续实际运行时的适配点：各脚本的 MongoDB 地址改为 `mongodb://127.0.0.1:27017`、日期窗口按本机数据覆盖调整、orderbook 的 1min 数据路径指向 `~/.qlib/qlib_data/cn_data_1min`。


## 六、qlib 环境为本批示例发生的变化（2026-10-05）

- 补装：`arctic`（--no-deps）、`pymongo==3.11.0`（由 4.17 降级，TaskManager 冒烟复验通过）、`lz4`、`tzlocal`、`tianshou==0.4.10`、`shimmy[gym]`、`fire`/`joblib`（arctic 环境补装）。
- arctic 专用 conda 环境保留（create_dataset.py 导入侧使用，pymongo 3.11 + py3.8）。
