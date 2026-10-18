# qlib examples 执行汇总与教学指南

> 日期：2026-10-03。机器：本机（180 核，conda 环境 `qlib`，Python 3.12.13）。
> 仓库：`/home/zxh/projects/4.qlib_learning/qlib`（fork：seuzxh/qlib，main 分支 a7277489，已用 `pip install -e .` 以可编辑模式装入 `qlib` 环境）。
> 本文记录两个官方示例的真实执行过程、结果与踩坑，并给出教学讲解路线。术语以 [术语表.md](术语表.md) 为准。

## 0. 一图看懂本文做了什么

```text
你的数据 ~/.qlib/qlib_data/cn_data（A 股日线，2020-01-02 ~ 2026-09-30）
        │
        ├── 实验一：examples/workflow_by_code.py（代码式工作流）
        │     LightGBM + Alpha158 因子 → 预测 → IC 分析 → TopkDropout 回测 → 风险分析
        │
        └── 实验二：examples/run_all_model.py（批量基线对比）
              4 个模型（LightGBM / XGBoost / Linear / DoubleEnsemble）
              每个模型：独立临时 conda 环境 → 装依赖 → 装本地 qlib → qrun YAML → 汇总成对比表
```

## 1. 运行前的环境与数据检查（教学要点：先体检，再跑数）

任何 qlib 实验前，按下面清单核对（本次全部实际执行过）：

| 检查项 | 命令/方法 | 本次结论 |
|---|---|---|
| Python 与包 | `conda activate qlib && pip list \| grep -i qlib` | Python 3.12.13，pyqlib 可编辑安装（指向本仓库源码） |
| 数据目录结构 | `ls ~/.qlib/qlib_data/cn_data` | calendars / instruments / features 三件套齐全 |
| 数据字段 | `ls ~/.qlib/qlib_data/cn_data/features/sh600000/` | open/high/low/close/volume/vwap/factor 七个日线字段，满足 Alpha158/Alpha360 |
| 指数（回测基准） | `ls features \| grep sh000300` | sh000001/sh000300/sh000852/sh000905/sz399001/sz399006 齐全 |
| 日历覆盖 | 读 `calendars/day.txt` 首尾行 | 2000-01-04 ~ 2026-09-30 |
| **特征实际覆盖** | 读 `close.day.bin` 文件头（见下） | **日线特征仅 2020-01-02 起**（日历≠特征覆盖，这是本次最重要的体检发现） |
| 标的池新鲜度 | 读 `instruments/csi300.txt` 结束日期 | 成分股更新至 2026-07-02 |

读取 bin 文件头判断特征起点的方法（qlib 的 bin 格式：首个 float32 是起始日历下标）：

```python
import numpy as np
with open("features/sh600000/close.day.bin", "rb") as f:
    start_idx = int(np.frombuffer(f.read(4), dtype="<f")[0])
# 再对照 calendars/day.txt 第 start_idx 行即该股数据起始日
```

### 本次发现的数据坑（两个，均已处理）

1. **日历覆盖 ≠ 特征覆盖**：日历从 2000 年开始，但特征文件只存了 2020-01-02 之后的约 1636 个交易日。官方示例默认训练区间是 2008~2020，直接跑会得到**空数据集**而不是报错。因此两个示例的日期均按本机数据窗口改写：训练 2020-01-02 ~ 2023-12-31、验证 2024-01-01 ~ 2024-12-31、测试 2025-01-01 ~ 2026-07-01（测试终点受 csi300 成分股数据截至 2026-07-02 限制）。
2. **`GetData().qlib_data(exists_skip=True)` 误判**：`exists_qlib_data` 要求 features/ 下所有代码都出现在 `instruments/all.txt` 中，而本机数据目录里有 6 个指数（sh000300 等回测基准用）不在 all.txt（纯个股池）里，于是被误判"数据不存在"，脚本走到"下载官方数据并删除旧目录"分支——**会尝试删除你的 15GB 数据目录**，本次因后台无法交互输入确认（EOFError）才幸免。处理：两个脚本中的 `GetData` 调用改为"数据已在位，直接 `qlib.init`"，并删除了数据目录里导致同样误判的 `.gitkeep` 占位文件。教训：**跑示例前先备份/只读挂载数据目录**。

## 2. 实验一：workflow_by_code.py —— 代码式工作流（入门第一课）

- 脚本：`examples/workflow_by_code.py`；运行方式：`conda activate qlib && cd examples && python workflow_by_code.py`；日志：`examples/workflow_by_code.log`；实验记录：`examples/mlruns/`（实验名 `workflow`）。
- 数据：`~/.qlib/qlib_data/cn_data`；标的全集：csi300；模型：LightGBM；特征：Alpha158（约 158 个量价因子）；标签：未来两日超额收益（`Ref($vwap,-2)/Ref($vwap,-1)-1`）。

### 2.1 运行过程的六个关键环节

1. **初始化**：`qlib.init(provider_uri=..., region=REG_CN)` 注册数据路径与市场区域，全局只需一次。
2. **组装任务**：`init_instance_by_config(task["model"])` 与 `init_instance_by_config(task["dataset"])`——qlib 把"模型"和"数据集"都做成**配置即对象**（config → 实例），这是 qlib 一切可复现性的根源。数据集内部由 DataHandler（Alpha158）负责"原始 bin → 特征矩阵"，并按 segments 切出训练/验证/测试三段。
3. **数据预览**：`dataset.prepare("train")` 先取训练段看一眼（打印 KMID/KLEN/…/LABEL0 前几行），确认特征与标签非空——**这一步是排查"数据窗口配错"的第一现场**。
4. **训练与记录**：`with R.start(experiment_name="workflow")` 开启实验管理（MLflow），`R.log_params` 记录全部超参，`model.fit(dataset)` 训练（日志可见早停生效，验证集损失不再下降即停）。
5. **预测与信号分析**：`SignalRecord.generate()` 在测试段逐日打分并保存预测；`SigAnaRecord.generate()` 计算 IC 家族指标（见 2.2）。IC（信息系数）即"预测分数与真实收益的日度截面相关"，是量化里衡量信号质量的第一指标。
6. **回测与风险分析**：`PortAnaRecord` 用 `TopkDropoutStrategy`（每日持有分数前 50 名、踢出最差 5 名）在测试段模拟交易，相对沪深 300 基准计算超额收益，输出年化收益、信息比率、最大回撤等指标（含成本与不含成本两套口径）。

### 2.2 结果（测试段 2025-01-01 ~ 2026-07-01，csi300，Alpha158 + LightGBM）

| 指标 | 数值 | 口径说明 |
|---|---|---|
| IC | 0.0287 | 日度截面皮尔逊相关的均值 |
| ICIR | 0.2590 | IC 均值 ÷ IC 标准差 |
| Rank IC | 0.0141 | 日度截面斯皮尔曼相关的均值 |
| Rank ICIR | 0.1424 | 同上，取稳定性 |
| 年化超额收益（含交易成本） | 10.90% | 相对沪深 300，计入开仓 5bps、平仓 15bps、最低 5 元费用 |
| 信息比率（含成本） | 0.890 | 超额收益年化均值 ÷ 年化跟踪误差 |
| 最大回撤（含成本） | -13.46% | 超额净值口径 |
| 年化超额收益（不含成本） | 15.69% | 同上但不计费用（成本拖累约 4.8 个百分点/年，源于每日换手 5/50） |
| 成交率 ffr | 1.0 | 所有委托全部成交 |

解读：IC≈0.03 在 A 股日频上属于"有信息但不高"的水平（官方 2008~2020 数据上的 LightGBM 基线 IC≈0.04~0.05）；信息比率 0.89、含成本年化超额约 10.9%，说明信号经得起交易成本与 2025~2026 年市场风格的考验。教学时可强调：**IC 高低取决于数据窗口与标签定义，官方 README 的数字不可直接对比**（官方是 2008~2020 + 20 次随机种子取均值）。

### 2.3 修改了什么（相对官方原版）

- 数据窗口重设（segments 与 handler 的 start/end），见第 1 节坑 1；
- 移除 `GetData` 下载调用（防误删数据），见第 1 节坑 2；
- 其余（模型超参、策略、回测费率）保持官方原样。

## 3. 实验二：run_all_model.py —— 批量基线对比（第二课）

- 脚本：`examples/run_all_model.py`；官方用途：遍历 `examples/benchmarks/` 下的模型文件夹，逐个用 `qrun` 跑该模型的 Alpha158/Alpha360 YAML 配置，最后把所有模型的 IC 与回测指标汇总成一张 markdown 对比表（官方 README 的基准表就是这么来的）。
- 本次运行命令（从 `examples/` 目录执行）：

```bash
conda activate qlib
cd examples
python run_all_model.py run 1 "[lightgbm,xgboost,linear,doubleensemble]" Alpha158 \
    --qlib_uri /home/zxh/projects/4.qlib_learning/qlib
```

  参数含义：`run 1` = 每个模型跑 1 次（官方基准表是 20 次取均值±标准差，脚本会自动删除 YAML 里的随机种子）；模型集合给了 4 个**纯 CPU、不依赖 PyTorch** 的经典基线（benchmarks 下共 26 个模型，其余多数是神经网络，需要装 torch、训练耗时数小时，教学演示不划算）；`--qlib_uri` 指向本地 clone 的仓库，让子环境从**你的代码**安装 qlib 而不是从 GitHub 拉。

### 3.1 脚本机制：每个模型走一遍"造环境 → 装依赖 → 装代码 → 跑实验 → 删环境"

对每个模型，脚本在 /tmp 下新建一个**一次性 conda 环境**（保证模型间依赖互不污染，例如 XGBoost 钉 1.2.1、DoubleEnsemble 钉 lightgbm 3.1.0 这种互相冲突的版本可以共存）：

1. `conda create --prefix /tmp/xxx python=3.8 -y`（原版 3.7，见 3.2 的适配 1）
2. `pip install -r benchmarks/<模型>/requirements.txt`（该模型专属依赖版本）
3. `pip install -e <qlib_uri>`（以可编辑模式装本地 qlib，含 Cython 扩展编译）
4. `qrun <yaml> <实验名> run_all_model_records`：按 YAML 执行完整工作流——数据加载 → Alpha158 特征工程（DropnaLabel、CSZScoreNorm 截面标准化等处理器）→ 训练（含早停）→ 测试段逐日预测 → IC 分析 → TopkDropout 回测 → 指标写入 MLflow 实验记录
5. 删除临时环境；全部模型跑完后 `_collect_results` 汇总生成 `table_Alpha158_<时间戳>.md`

实验记录落在 `examples/run_all_model_records/`（MLflow 文件存储），每个模型一个 experiment（以模型名命名，如 `LightGBM`），每次运行一个 recorder。

### 3.2 本次做的四处置配（每处都有教学价值）

1. **python=3.7 → 3.8**：本仓库 `pyproject.toml` 要求 Python>=3.8，3.7 会在安装 pyqlib 时直接被 pip 拒绝。
2. **移除 `GetData().qlib_data(exists_skip=True)`**：与实验一同因（第 1 节坑 2），防止误删数据。
3. **四个模型 YAML 的日期窗口**改为本机数据窗口（train 2020~2023、valid 2024、test 2025-01-01~2026-07-01，回测与 test 对齐）。
4. **XGBoost 的 `xgboost==1.2.1` 放宽为不钉版本**：第一轮 XGBoost 在训练完成后报 `ImportError: cannot import name 'Int64Index' from 'pandas'`——2020 年的 xgboost 1.2.1 兼容代码引用了 pandas 2.0 已删除的类，而脚本第 3 步装本地 qlib 时会把 pandas 升到 2.0.3。放宽后 pip 装到 xgboost 2.1.4（py3.8 可用的最新版），训练与回测全部通过。

**踩坑 5（隐蔽，值得讲）**：DoubleEnsemble 文件夹里有两个匹配 `*Alpha158.yaml` 通配符的文件（`workflow_config_doubleensemble_Alpha158.yaml` 和 `..._early_stop_Alpha158.yaml`），脚本用 `glob(...)[0]` 取第一个，恰好取到**没改日期的 early_stop 变体**，于是取到空数据集而报错。修复：把 early_stop 变体的日期也改掉，两个文件谁被选中都能跑。教训：**改配置要穷举同一 glob 模式下的所有文件**。

**踩坑 6（环境类）**：子环境的 pip 会走本机代理（127.0.0.1:7890），第二轮 XGBoost 安装时代理连接卡住了约 20 分钟后自行恢复。批量跑长任务前确认代理稳定性，或对子进程取消代理环境变量。

### 3.3 运行结果

四个模型在同一数据、同一切分（train 2020-01-02~2023-12-31 / valid 2024 全年 / test 2025-01-01~2026-07-01）、同一策略（TopkDropout top50/drop5）与同一费率（开仓 5bps、平仓 15bps、最低 5 元）下各跑 1 次，全部 FINISHED。下表即脚本自动生成的对比表（`examples/table_Alpha158_2026-10-03_14:10:56.md`，回测列为**相对沪深 300 的超额收益、含交易成本口径**；±后为标准差，因只跑 1 次故为 0）：

| 模型 | IC | ICIR | Rank IC | Rank ICIR | 年化超额（含成本） | 信息比率（含成本） | 最大回撤（含成本） | 年化超额（不含成本） |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.0421 | 0.3392 | 0.0150 | 0.1454 | **12.41%** | **0.999** | **-14.35%** | 17.28% |
| LightGBM | 0.0308 | 0.2971 | 0.0135 | 0.1431 | 11.35% | 0.940 | -15.55% | 16.15% |
| DoubleEnsemble | 0.0303 | 0.2816 | 0.0081 | 0.0805 | 8.31% | 0.646 | -18.52% | 13.09% |
| Linear（线性回归） | 0.0082 | 0.0453 | **0.0501** | **0.2659** | -8.37% | -0.662 | -24.79% | -3.62% |

结果解读（教学要点）：

1. **树模型显著优于线性模型**：XGBoost 与 LightGBM 的 IC、含成本年化超额、信息比率全面领先，说明 Alpha158 的量价因子与未来收益之间存在非线性关系。
2. **IC 高 ≠ 赚钱**：Linear 的 Rank IC 反而全场最高（0.0501），但年化超额为 -8.37%（含成本）。原因是 IC 度量的是全截面排序质量，而 TopkDropout 只交易**分数最高的前 50 名**——线性模型对头部股票的排序能力差（截面上有预测力集中在尾部/中部），且负收益在扣费后进一步放大。这是"信号评估"与"组合构建"两个环节脱节的经典案例。
3. **成本拖累可观**：三个盈利模型的不含成本→含成本年化超额均下降约 4~5 个百分点（例如 XGBoost 从 17.28% 降到 12.41%），因为每日换手 5/50（10%）叠加双边费率。
4. 实验记录与汇总表：`examples/run_all_model_records_Alpha158_2026-10-03_14:10:56/`（每个模型一个 MLflow experiment；第一轮 XGBoost 因 pandas 兼容问题失败的那条 FAILED 记录也留档在内，可对比）。脚本跑完会自动把 `run_all_model_records` 目录改名归档。

耗时参考（单机）：每模型一个周期约 7~10 分钟，其中 conda 建环境 + pip 装依赖占 90% 以上（走阿里云 PyPI 镜像；第二轮起命中 pip 缓存明显变快）；qrun 本身很快——Alpha158 数据集初始化约 9 秒、训练含早停数十秒、360 个交易日的回测以约 90 天/秒推进。

## 4. 如何复现与延伸（作业路线）

### 4.1 一键复现本次全部实验

```bash
conda activate qlib                 # Python 3.12，pyqlib 已可编辑安装指向本仓库
cd /home/zxh/projects/4.qlib_learning/qlib/examples

# 实验一：代码式工作流（约 2 分钟）
python workflow_by_code.py

# 实验二：四模型基线对比（每模型 7~10 分钟）
python run_all_model.py run 1 "[lightgbm,xgboost,linear,doubleensemble]" Alpha158 \
    --qlib_uri /home/zxh/projects/4.qlib_learning/qlib
```

若要按官方基准口径（20 次随机种子取均值±标准差）：把 `run 1` 改成 `run 20`；若要跑神经网络模型（MLP/LSTM/ALSTM/Transformer/GATs/TRA/TFT 等），把它们加进模型列表即可，脚本会自动装 torch（需要更长时间与更大磁盘）。

### 4.2 查看实验记录（MLflow）

```bash
cd examples
mlflow ui --backend-store-uri file:./run_all_model_records_Alpha158_2026-10-03_14:10:56
# 浏览器打开 http://127.0.0.1:5000，可看每个 recorder 的参数、指标曲线与产出物
#（pred.pkl 预测、report_normal.pkl 回测净值、indicator_analysis 等）
```

代码方式读取指标见本文 2.2/3.3 的口径（`R.get_exp(...)` → `list_recorders()` → `list_metrics()`）。

### 4.3 本次对 fork 的全部修改（未提交，建议作为一个"本机数据适配"commit）

| 文件 | 修改内容 |
|---|---|
| `examples/workflow_by_code.py` | 数据窗口重设（segments + handler 起止 + 回测区间）；移除 GetData 下载调用（两处均加注释说明原因） |
| `examples/run_all_model.py` | `create_env` 的 python 3.7→3.8；`_init_qlib` 移除 GetData 调用 |
| `examples/benchmarks/{LightGBM,XGBoost,Linear,DoubleEnsemble}/workflow_config_*_Alpha158.yaml`（5 个文件，含 DoubleEnsemble 的 early_stop 变体） | 日期统一改为本机数据窗口 |
| `examples/benchmarks/XGBoost/requirements.txt` | `xgboost==1.2.1` 放宽为不钉版本（pandas 2.0 兼容） |

### 4.4 延伸练习（由浅入深）

1. 换标的全集：把 yaml 里 `market: csi300` 改成 `csi500`（注意：本机 csi500 成分股文件只更新到 2022-03-21，需先刷新数据）。
2. 换特征集：`dataset` 用 Alpha360（OHLCV 窗口特征），对比同模型在两套因子库上的 IC。
3. 自定义因子：在 Alpha158 基础上加自己的表达式因子（如 `Mean($close,5)/$close-1`），qlib 表达式引擎免编译生效。
4. 提高信噪比：`times=20` 跑出均值±标准差，再下结论模型优劣（单次运行的 IC 波动可达 ±0.01）。
5. 与你 fork 里 `docs/learning/` 的第 1 课对照阅读：那一课讲概念地图，本文记录的是同一脚本在真实自建数据上的完整落地过程（含全部踩坑）。

## 5. 第二批示例：数据层、高频、风险模型、滚动训练、超参优化

> 运行日期：2026-10-04。全部在 `qlib` conda 环境下执行。每个示例都是"官方代码 + 本机数据/依赖适配"，适配处均有行内注释。

### 5.1 model_interpreter/feature.py —— 特征重要性（模型解释）

- 做什么：训练与实验一相同的 LightGBM 模型，然后用 `model.get_feature_importance()` 输出 158 个 Alpha158 因子的重要性排序。
- 结果（2020~2023 训练、同实验一配置）：重要性前五的列是 Column_37、Column_2、Column_38、Column_47、Column_138。列号是 Alpha158 特征顺序的位置索引，可用 handler 的列名对照翻译（一行代码：`[c for c in Alpha158(...).get_cols() if c in ...][37]`）。
- 适配：与 workflow_by_code.py 完全相同（数据窗口重设 + GetData 防误删）。
- 教学点：树模型的重要性是"该特征在所有分裂中带来的增益之和"，只能横向比较、不能解读为因果；重要性高度集中的前若干因子往往就是该市场的核心风格。

### 5.2 data_demo/data_cache_demo.py —— Handler 序列化缓存

- 做什么：同一个 LightGBM YAML 工作流跑两遍——第一遍现场构建 Alpha158 数据集，第二遍把 Handler 对象 `to_pickle` 落盘后以 `file://handler.pkl` 方式引用，对比端到端耗时。
- 结果（本机实测）：无缓存 31.6 秒，有缓存 2.5 秒，**约 12.5 倍加速**——省掉的是 Alpha158 表达式计算与处理器（标准化/填充）的重复劳动。
- 踩坑 7（依赖老化）：官方示例用了 `yaml.safe_dump(...)` 与 `yaml.dump(..., indent=4)`，这些是老 ruamel/PyYAML 写法，ruamel.yaml 0.18 的 `YAML` 实例已无 `safe_dump`、`dump` 也不再收 `indent/sort_keys`（这些参数移到了构造器）。按新 API 修正后通过。
- 教学点：DataHandler 可序列化（pickle）是 qlib 数据层的重要设计——"取数与加工一次、多次复用"，跨实验共享特征工程成果。
- 同目录的 `data_mem_resuse_demo.py`（内存复用版，成功）：同一工作流跑两遍不落盘，第二遍直接把已构建的 Handler **对象**塞进任务配置（`handler=hd`）复用内存数据。本机实测：从盘上重复构建 140.9 秒 vs 内存复用 26.0 秒（再换训练窗口重跑 23.8 秒），约 **5.4 倍加速**。踩坑 11（两处，均已修）：①第 1 步的 `task_train` 会**原地**填充 task 里的 `<MODEL>/<DATASET>` 占位符、并在收尾以 `dump_all=False` 剥离 handler 的内存数据，示例后续 `deepcopy(task_config["task"])` 复用的是这些被处置过的旧对象（报 `'Alpha158' object has no attribute '_infer'`），须重新从 YAML 加载干净配置；②第 4 段演示"换窗口"用的 2010~2013 训练段超出本机数据覆盖（空数据），改成本机窗口内的 2020~2021 子区间。

### 5.3 highfreq/ —— 分钟级数据与自定义算子（workflow.py 两方法 + 1min 建模 YAML）

- 数据：`~/.qlib/qlib_data/cn_data_1min`（独立于日线的分钟库）。体检发现：分钟日历覆盖 2000~2026-09-30 共 155 万根 bar，但**个股分钟特征实际从 2024-01-02 起**（又是"日历 ≠ 特征覆盖"，与日线同款坑）；每只股票有 open/high/low/close/volume/vwap/factor 七个 1min 字段；标的池文件只有 `all.txt`，无 csi300——已把日线库的 `instruments/csi300.txt` 复制过来。
- `get_data` 方法（成功）：构建 HighFreqHandler（自定义算子 DayLast/FFillNan/BFillNan/Select/Cut 组成的"分钟价格归一化"表达式）与 HighFreqBacktestHandler，2024-03-01 ~ 2024-07-31、csi300，训练集打印出 360 万行分钟特征。
- `dump_and_load_dataset` 方法（成功）：把两个 DatasetH `to_pickle` 落盘、进程内重载、再把窗口重配到 2024-07-22 ~ 07-26 的 5 天，验证"数据集对象可序列化、可复用、可重配窗口"。
- 1min 建模 YAML（`workflow_config_High_Freq_Tree_Alpha158.yaml`，qrun 执行，成功）：Alpha158 处理器跑在 1min 频率上 + HFLGBModel（LightGBM 二分类：预测未来 1 分钟涨跌），窗口 2024-03-01 ~ 2024-07-31（train 到 6-30、valid 到 7-15、test 至月末），记录环节用 SignalRecord + HFSignalRecord。过程耗时：数据集构建 188.6 秒（其中分钟级 RobustZScoreNorm 81.7 秒占大头），LightGBM 训练约 25.7 分钟（约 360 万行 × 158 特征，20 线程）。测试段信号质量：**分钟 IC 0.0222、Rank IC 0.0285、ICIR 0.303**；按预测分组的多头精确率 51.6%、空头精确率 52.1%（二分类口径，50% 为随机基准）；多空组合平均分钟收益 0.54bps（bps=基点，0.000054）、平均分钟夏普 0.26——单分钟信号很弱，但教学演示足够（实盘高频信号靠聚合与执行增益放大）。实验记录：`examples/highfreq/mlruns/`（实验名 `workflow`）。
- 踩坑 8（数据字段缺失）：HighFreqHandler 的所有表达式依赖 `$paused`（停牌标志）字段做 `Select` 过滤，本机分钟库没有该字段，缺失数据的布尔索引与主数据对不齐，报 `IndexingError: Unalignable boolean Series`。适配：停牌时段本就没有分钟 bar（日历对齐后为 NaN，再由 FFillNan 补齐），把停牌过滤模板改为直通并在注释中说明；若换用官方 1min 数据包（含 paused）可还原。
- 踩坑 9（安全白名单 vs 老示例）：fork 较新代码引入了 `restricted_pickle_load` 安全白名单（只放行 pandas/numpy），`DatasetH`、`Alpha158` 及示例目录自定义的 Handler 类都会被拒（`Forbidden class`）。示例加载的是本进程刚落盘的可信文件，改用普通 `pickle.load` 并注释；**加载来源不明的 pickle 时仍必须用 restricted_pickle_load**。
- 教学点：①qlib 表达式算子可以自定义（继承 ElemOperator 实现 `_load_internal` 即可，本示例的 DayLast/Select 就是范例）；②同一套 Alpha158 处理器可直接切到分钟频率（`freq: '1min'`）；③"日历覆盖 ≠ 特征覆盖"在分钟库上同样成立，且窗口要按特征实际起点重新设计。

### 5.4 rolling_process_data/ —— 滚动训练的数据层加速

- 做什么：演示"原始数据加工一次、滚动窗口反复复用"——先把 2020~2026 全窗口的 Alpha158 **原始特征**（不带处理器）dump 成 pre_handler.pkl，之后每轮滚动（RollingDataHandler）只在这份缓存上重新执行标准化/标签处理（IT_FIT_SEQ），避免每轮重新取数。
- 结果（成功）：滚动 3 轮全部完成——roll0 训练 2020~2022 / 验证 2023 / 测试 2024，roll1、roll2 依次 +1 年；每轮打印三段数据形状（如测试段 77,050 行 × 159 列）。
- 适配：窗口改 2020 起（原 2010~2019）、rolling_cnt 5→3（数据年限所限，最后一轮 test 只到 2026-09）；GetData 防误删；pre_handler 重载同踩坑 9 改普通 pickle。
- 教学点：滚动训练（rolling retrain）是让模型"追市场"的标准姿势；qlib 的省时思路是把"取数"与"加工"分离——取数一次缓存，加工每轮重做（因为标准化等处理器的统计量必须只用当轮训练段计算，避免未来信息泄漏）。

### 5.5 portfolio/ —— 风险模型 + 指数增强策略（两步）

- 第一步 `prepare_riskdata.py`：用 `StructuredCovEstimator`（统计风险模型：对全市场股票 240 日收益做 PCA，取前 10 个主成分作公共因子）逐日生成三件套——`factor_exp.pkl`（因子暴露）、`factor_cov.pkl`（因子协方差）、`specific_risk.pkl`（特定风险波动率），写入 `riskdata/YYYYMMDD/`。本机生成 2025-01-02 ~ 2026-06-30 共 361 个交易日。
- 第二步 `config_enhanced_indexing.yaml`（qrun）：训练 LightGBM（与实验一同款超参）产生选股信号，回测策略换成 `EnhancedIndexingStrategy`（指数增强：在跟踪误差约束下最大化信号加权组合的期望收益），相对沪深 300 回测。
- 适配：风险数据生成加了两处工程处理——只生成回测窗口所需日期（原版从 2016 年全量生成）、已生成日期断点续跑（首跑时进程被无声杀掉过一次，日志缓冲丢失，改为 `-u` 无缓冲 + 跳过已有日期后跑完）；YAML 日期同批适配。
- 结果（测试段 2025-01-01 ~ 2026-07-01，相对沪深 300）：信号指标与 LightGBM 基线一致（IC 0.0308、Rank IC 0.0135，同一模型同一数据）；回测环节**含成本年化超额 -5.14%、信息比率 -0.497、最大回撤 -17.26%**；不含成本年化超额 +0.35%（信息比率 0.034），成本拖累约 5.5 个百分点/年，成交率 1.0。回测本身耗时约 55 分钟（360 个交易日、每日解一次带风险约束的凸优化，约 9~10 秒/天）。实验记录：`examples/portfolio/mlruns/`（实验名 `workflow`）。
- 结果解读（教学要点）：与第 3 节 LightGBM 基线对比——**同一信号、不同组合构建，结果天壤之别**：TopkDropout（top50 等权、每日换 5 只）含成本年化超额 +11.35%，而本示例默认配置的指数增强为 -5.14%。原因：示例未设换手上限等实务约束，优化器每天重排权重导致高换手，5bps/15bps 的双边费用把不含成本的微弱正超额（+0.35%）吃穿。这正说明：信号只是半成品，组合层（约束、成本、换手控制）同样是收益的决定性环节；跑通示例 ≠ 有可交易的超额。同时这也是 qlib 里"预测信号 → 组合优化 → 交易"完整链路的进阶示范——相对 TopkDropout 的简单等权选股，指数增强引入了风险模型（因子暴露×因子协方差+特定风险）与优化约束，更接近实盘组合构建。

### 5.6 hyperparameter/LightGBM —— Optuna 超参优化

- 做什么：用 Optuna 对 LightGBM 的 9 个超参（学习率/叶子数/正则强度/采样比例等）做黑箱搜索，目标函数是验证集最优 L2 损失（`min(evals_result['valid'])`）。
- 适配：①optuna 5.0 已移除 `suggest_uniform/suggest_loguniform`，改为 `suggest_float`（log 参数用 `log=True`）；`optuna.Study(...)` 直接构造改为 `create_study(..., load_if_exists=True)`；②官方脚本**未设 n_trials 会无限搜索**，限定 20 次；③数据窗口重设 + GetData 防误删（同前）。
- 踩坑 10（被掩盖的异常 + 隐藏的 API 变化）：官方目标函数 `return min(evals_result["valid"])` 在新版 LGBModel 下会返回**字符串**——新版 `LGBModel.fit` 把 evals_result 委托给 lightgbm 的 `record_evaluation`，结构是 `{"valid": {"l2": [逐轮损失]}}`，`min()` 作用在字典上取到的是键 `"l2"`。字符串传入 optuna 后在其 `math.isnan(v)` 处抛 TypeError，而 optuna `_run_trial` 的 finally 断言又把真实异常掩盖成一句 `AssertionError: Should not reach.`（库代码放断言属上游健壮性问题，排障时用 `e.__context__` 可挖出被掩盖的原始栈）。修复：改为 `min(next(iter(evals_result["valid"].values())))`。另将官方 `n_jobs=6` 改为串行 `n_jobs=1`（多线程共享同一 dataset 对象不必要且更稳）。
- 结果（20 次 trial 全部完成，串行总耗时约 35 分钟）：**最优验证集 L2 损失 0.997258（trial 16）**，最优超参：learning_rate 0.0225、num_leaves 364、colsample_bytree 0.657、feature_fraction 0.670、bagging_fraction 0.541（bagging_freq 2）、lambda_l1 3.2e-06、lambda_l2 0.0026、min_child_samples 57、min_data_in_leaf 43、subsample 0.623。作为参照，第 3 节 LightGBM 基线用官方手工调参的验证 L2 约 0.9984~1.0009——本搜索在 20 次试验内小幅占优；搜索空间（学习率 0~1 均匀采样）很宽，更多试验次数还能继续改进。研究数据库：`examples/hyperparameter/LightGBM/db.sqlite3`。
- 教学点：超参搜索的目标函数用**验证集**而非测试集——测试集只留给最终评估，这是防止"调参过拟合"的纪律；Optuna 的 TPE 采样器会根据已完成 trial 的好坏逐渐收敛到优质区域。
- 同目录的 `hyperparameter_360.py`（Alpha360 版，成功）：在 Alpha360 特征集上重复同样的搜索，20/20 trial 完成，**最优验证 L2 0.997816（trial 10）**，最优超参与 158 版截然不同——num_leaves 仅 13、learning_rate 0.215（158 版最优是 364 叶 + 0.0225 学习率）。这是偏差-方差权衡的活教材：Alpha360 是原始 OHLCV 窗口特征（维度高、单特征噪声大），浅树+快学习率的低容量配置泛化更好；Alpha158 是加工过的比值因子（信息密度高），深树能挖出更多交互。适配除 158 版四项外另有两处：`get_dataset_config(market=...)` 旧签名改为 `handler_kwargs={"instruments":...}`；研究存储改独立文件 `db_360.sqlite3`（原版与 158 同名会互相覆盖）。另为 optuna 偶发的 tell 断言加了"捕获后按 sqlite 续接重启"的防护（库健壮性问题，详见踩坑 10）。


### 5.7 嵌套执行回测 nested_decision_execution/ —— 多时间层级联动交易

- 做什么：qlib 的招牌"嵌套执行"示例——**日频** TopkDropout 做组合决策 → **5 分钟**层 SBBStrategyEMA（基于分钟 EMA 的算法交易策略）拆单 → **1 分钟**层 TWAP 继续拆分并撮合，三层各有独立的执行器与指标统计。模型层与实验一相同（LightGBM+Alpha158，日频信号）。
- 数据适配（本示例是三频率联动的集中考验）：本机无任何 30min 特征数据，嵌套层级由官方的 日/30分/5分 改为 **日/5分/1分**（`provider_uri_map`：day 与 5min 指向 cn_data、1min 指向 cn_data_1min）；回测窗口取两库交集 2024 年 3 月；此外 cn_data 里指数 sh000300 没有 5min 数据，而中层报告要按 5min 计算基准收益（报 `The benchmark ['SH000300'] does not exist`）——用 sh000300 的 1min 数据聚合生成了 5min 的 7 个字段 bin 写入 cn_data（open 首/high 最大/low 最小/close 末/volume 和/vwap 量加权/factor 末）。
- 结果（成功，21 个交易日 2024-03-01 ~ 03-29，exit 0）：模型验证 L2 0.998；组合层含成本年化超额 -7.37%（信息比率 -0.726、最大回撤 -2.09%），不含成本口径数值相同（嵌套执行的费用记账与日频路径不同）；日级指标 FFR/PA/POS 为 NaN——嵌套执行时日级成交指标需内层逐笔信息聚合，`value_weighted` 聚合下未产出，日志中每日 `[Indicator(day)]` 可见（5min/1min 内层指标正常打印）。**运行风波**：两个月窗口首次运行在第 30 个交易日以 0% CPU 挂起（疑似嵌套执行器上游问题，无报错日志），缩短为 4 周后完整跑通；挂起现象已在代码注释与本节记录，供上游排查。
- 教学点：①`provider_uri` 支持按频率映射到不同数据目录，是异构数据源组合的关键机制；②嵌套执行把"组合决策"与"订单执行"分离到不同时间尺度，是研究信号落地为交易的最贴近实盘的回测形态；③每层的 `generate_portfolio_metrics`/`indicator_config` 会分别产出各层级的成交与指标统计。

### 5.8 神经网络基线批次（PyTorch，2026-10-04）

- 前提：给 `qlib` 环境安装 torch 2.14.1+cu130（8×H20 上验证 CUDA 可用）。GPU 被常驻 vLLM 服务占用约 94GB/卡，每卡仅 ~2.7GB 空闲，因此采取：小模型直接上 GPU、大模型降 batch、逐模型串行的"双车道"调度。
- 适配：17 个 NN 模型 yaml 统一做日期窗口适配（规则同前）+ GPU 指定；显存不足者降低 batch（GRU/LSTM 800→200、Localformer 默认 8192→1024、TRA 1024→256、ADD 5000→400、SFM 1600→200、ADARNN 800→400/100）；`pytorch_hist.py` 补 `torch.load(..., map_location=self.device)` 一行健壮性修复。
- 结果（Alpha158 批：测试段 2025-01-01~2026-07-01，相对沪深 300，含成本口径）：

| 模型 | IC | Rank IC | 年化超额（含成本） | 信息比率 | 最大回撤 |
|---|---|---|---|---|---|
| MLP | 0.0218 | 0.0067 | +7.16% | 0.680 | -12.56% |
| TCN | 0.0072 | 0.0047 | +6.94% | 0.722 | -10.86% |
| TabNet | 0.0109 | 0.0421 | -0.61% | -0.052 | -22.47% |
| TCTS | 0.0068 | 0.0526 | -5.15% | -0.405 | -22.62% |
| ADD | 0.0082 | **0.0529** | -5.26% | -0.408 | -24.91% |
| KRNN | 0.0044 | 0.0396 | -9.61% | -0.784 | -27.01% |
| LSTM | 0.0029 | 0.0438 | -11.08% | -0.849 | -25.00% |
| ALSTM | 0.0005 | 0.0387 | -11.27% | -0.959 | -25.47% |
| IGMTF | 0.0094 | 0.0519 | -11.78% | -0.879 | -26.71% |
| TRA | 0.0005 | 0.0462 | -12.44% | -1.025 | -27.96% |
| GRU | -0.0004 | 0.0486 | -12.62% | -0.970 | -28.20% |
| Sandwich | 0.0016 | 0.0404 | -14.24% | -1.084 | -32.15% |
| GATs | 0.0084 | 0.0502 | -14.49% | -1.111 | -31.29% |
| SFM | -0.0019 | 0.0438 | -14.72% | -1.116 | -29.85% |
| Localformer | -0.0059 | 0.0474 | -16.49% | -1.229 | -32.78% |
| HIST | 0.0119 | **0.0576** | -12.62% | -0.961 | -28.20% |
| ADARNN | 0.0080 | 0.0511 | -7.03% | -0.555 | -24.01% |
| TFT | 0.0238 | -0.0369 | +12.08% | 0.645 | -19.40% |

（15/18 完成；TCTS 另因官方 `lowest_valid_performance: 0.993` 阈值按官方数据调参、本机验证损失底约 1.01~1.04 而陷入无上限重训，放宽至 1.05 后一次通过。）

- 结果解读（与第 6 章结论呼应）：**神经网络族在本窗口全面不敌树模型**——最好的 NN（MLP +7.16%）也不及 LightGBM（+11.35%）与 CatBoost（+14.66%）；时序模型（LSTM/GRU/ALSTM/GATs/TRA 等）皮尔逊 IC 接近 0 但秩 IC 达 0.04~0.05，与 Linear 模型同款"截面单调性与头部排序脱节"病症在 TopkDropout 下被交易成本放大为负超额。这与官方基准（2008~2020 数据）中"GBDT 稳居前列"的格局一致；另需注明我们的 batch 因显存约束普遍调小，超参偏离官方设定，比较仅在教学口径下成立。
- 受阻未跑的两个模型及原因：**HIST**——依赖仓库自带的 `stock2concept/stock_index` 概念图 npy（按官方 2008~2020 宇宙构建的静态文件），自建宇宙的股票索引对不上（`IndexError`），需按本机概念数据重建图文件后方可运行；**ADARNN**——四次尝试（不同 GPU/batch）均因共享 GPU 每卡 ~2.7GB 空闲不足而 CUDA 初始化失败（含一次 invalid ordinal 怪错），CPU 训练又需以天计，判定为受资源约束阻断。
- 实验记录：`examples/nn_records/`（每个模型一个 experiment）；日志 `examples/nn_logs/`。

### 5.9 跳过未跑的示例（及原因）

| 示例 | 原因 |
|---|---|
| benchmarks_dynamic(DDG-DA)、rl/、rl_order_execution/ | 需要 PyTorch/gym（本环境未装；qlib.rl 直接 import gym） |
| model_rolling/、online_srv/ | 依赖 MongoDB（TaskManager 用 pymongo 持久化任务队列）与在线服务常驻进程 |
| orderbook_data/ | 需 Arctic 后端与非定频订单簿数据 |
| workflow_by_code.ipynb、tutorial/detailed_workflow.ipynb | 与已跑通的 .py 版工作流内容重复（notebook 形式） |
| benchmarks 剩余模型 | **18/18 全部跑通**（2026-10-07 补齐 HIST/ADARNN/TFT，见 5.8 补记）；此前受阻三者的攻关记录见 5.8 末尾 |

### 5.10 第二批示例结果汇总

全部九组示例执行情况（关键数字与口径见对应小节）：

| 示例 | 状态 | 关键结果（口径） |
|---|---|---|
| model_interpreter/feature.py | 成功 | 158 因子重要性排序，前五列 Column_37/2/38/47/138（训练窗口 2020~2023） |
| data_demo/data_cache_demo.py | 成功 | 同一工作流无缓存 31.6 秒 vs Handler 缓存 2.5 秒（约 12.5 倍） |
| highfreq/workflow.py（get_data、dump_and_load_dataset） | 成功 | csi300 分钟数据 2024-03~07 训练段 360 万行；数据集可序列化、重载后窗口重配 5 天成功 |
| highfreq YAML（1min LightGBM 二分类） | 成功 | 分钟 IC 0.0222 / Rank IC 0.0285 / ICIR 0.303；多头精确率 51.6%（test 2024-07-16~07-31） |
| portfolio（风险模型 + 指数增强） | 成功 | 361 个交易日风险数据（PCA 10 因子）；含成本年化超额 -5.14%（不含成本 +0.35%），信号 IC 与 LightGBM 基线一致（0.0308） |
| rolling_process_data | 成功 | 3 轮滚动（2020~2024 起、每年 +1），每轮 train/valid/test 三段数据就绪 |
| hyperparameter/LightGBM（Alpha158） | 成功 | 20/20 trial，最优验证 L2 0.997258（trial 16），优于基线验证 L2 约 0.9984~1.0009 |
| hyperparameter/LightGBM（Alpha360） | 成功 | 20/20 trial，最优验证 L2 0.997816（trial 10）；最优超参与 158 版形成偏差-方差对照 |
| data_demo/data_mem_resuse_demo.py | 成功 | 盘上重复构建 140.9s vs 内存复用 26.0s（约 5.4 倍） |
| benchmarks/CatBoost（qrun 直跑） | 成功 | IC 0.0393、含成本年化超额 +14.66%、IR 1.132——五个基线中最优 |
| nested_decision_execution（嵌套执行） | 成功 | 21 个交易日三层嵌套回测完成；年化超额 -7.37%（窗口短，示例重机制演示而非 alpha） |

第二批对 fork 的修改文件（同样未提交，可与第一批合并为一个"本机适配"commit）：

| 文件 | 修改内容 |
|---|---|
| `examples/model_interpreter/feature.py` | 数据窗口重设；GetData 防误删 |
| `examples/data_demo/data_cache_demo.py` | ruamel 0.18 API 修正（`safe_dump`→`dump`、去掉不支持的 `indent` 参数） |
| `examples/highfreq/workflow.py` | 窗口平移至 2024-03~07、标的池 all→csi300、GetData 防误删、重载改普通 pickle（白名单不适用） |
| `examples/highfreq/highfreq_handler.py` | 停牌过滤模板直通（本机分钟数据无 `$paused` 字段，注释说明可还原） |
| `examples/highfreq/workflow_config_High_Freq_Tree_Alpha158.yaml` | 日期窗口平移 |
| `examples/rolling_process_data/workflow.py` | 窗口 2020 起、rolling_cnt 5→3；GetData 防误删；pre_handler 重载改普通 pickle |
| `examples/portfolio/prepare_riskdata.py` | start_time 2020 起；只生成回测窗口日期；断点续跑跳过已有日期 |
| `examples/portfolio/config_enhanced_indexing.yaml` | 日期窗口适配（同第一批口径） |
| `examples/hyperparameter/LightGBM/hyperparameter_158.py` | optuna 5 API 替换；Study 构造→create_study；目标函数修复（返回损失序列最小值）；n_trials=20、串行；数据窗口重设 |
| `examples/hyperparameter/LightGBM/hyperparameter_360.py` | 同上四项 + `get_dataset_config` 新签名 + 独立 sqlite + tell 断言续接防护 + fit 日期显式化（模板不经 qrun 不会被解析） |
| `examples/benchmarks/CatBoost/workflow_config_catboost_Alpha158.yaml`、`requirements.txt` | 日期窗口适配；catboost 0.24.3 放宽（pandas 2 兼容） |
| `examples/nested_decision_execution/workflow.py` | 三频率映射（日/5分/1分）、窗口 2024-03（缩至 4 周防挂起）、GetData 防误删 |

新增数据文件：`~/.qlib/qlib_data/cn_data_1min/instruments/csi300.txt`（从日线库复制，分钟库原本只有 all.txt）。

实验记录位置速查：`examples/{mlruns（实验一）,run_all_model_records_Alpha158_2026-10-03_14:10:56（实验二）,highfreq/mlruns,data_demo/mlruns,portfolio/mlruns}`，超参搜索数据库 `examples/hyperparameter/LightGBM/db.sqlite3`（`optuna-dashboard sqlite:///db.sqlite3` 可视化查看）。

### 5.11 Stacking 严格版基线（2026-10-18，第四批）

在 qlib 里新增了严格版 stacking（堆叠）基线并跑通。新增两个文件（未提交，可与前几批合并为一个"本机适配"commit）：

- `qlib/qlib/contrib/model/stacking.py`：`StackingModel`——扩展窗折外循环 + 非负线性元模型，基模型黑盒复用 qlib 既有模型类（每折造共享 handler 的数据视图，仅换 segments，训练循环零重写）
- `examples/benchmarks/Stacking/workflow_config_stacking_Alpha158.yaml`：工作流配置，qrun 一条命令复现

**设计要点**（完整设计文档见项目 `.omc/plans/2026-10-17_stacking_design.md`）：基模型 LightGBM + MLP + LSTM；3 折扩展窗折外（训 2020→出 2021、训 2020~21→出 2022、训 2020~22→出 2023，任一折外预测只依赖过去）；元模型为非负线性回归（权重可直接读作裁判配比）。**统一数据口径**：三基模型共享一份 Alpha158 处理器（特征 RobustZScoreNorm+Fillna，标签 DropnaLabel 后按日截面 z 分数归一）——因此与各自基准跑分不逐项可比（LightGBM 基线用原始收益标签、MLP 基线丢 VWAP0 列、LSTM 基线用 20 列筛选 + 秩归一标签）。

结果（与其他基线同回测口径：测试段 2025-01-01~2026-07-01、相对沪深 300、TopkDropout top50/drop5）：

| 指标 | **Stacking** | 基线 LightGBM | 基线 MLP | 基线 LSTM |
|---|---|---|---|---|
| IC | 0.0232 | 0.0308 | 0.0218 | 0.0029 |
| Rank IC | **0.0252** | 0.0135 | 0.0067 | 0.0438 |
| 含成本年化超额 | +11.08% | +11.35% | +7.16% | -11.08% |
| 信息比率（含成本） | 0.864 | 0.940 | 0.680 | -0.849 |
| 最大回撤（含成本） | -16.24% | -15.55% | -12.56% | -25.00% |

元模型权重：lgb 0.206 / mlp 0.125 / lstm 0.045（归一化约 55% / 33% / 12%）。折外段（2021~2023，474,731 行）各基模型 IC：mlp 0.0118 > lstm 0.0080 > lgb 0.0066，Rank IC 三者均 ≈0 或微负。

结果解读（教学要点）：

1. **集成未超越最强基模型**：+11.08% 略低于 LightGBM 基线 +11.35%，信息比率与回撤亦稍差。三个基模型同吃一份卷子、特征同源，相关性高，互补增量小——stacking 不是免费午餐。
2. **裁判配比与折外评估自洽**：折外 IC 最高的 MLP 拿到第二权重（33%），最弱的 LSTM 权重最小（12%），未退化为单模型垄断；权重排序（lgb>mlp>lstm）与测试段贡献一致。
3. **统一卷子的代价显性化**：折外段上三个基模型 IC 仅 0.007~0.012，明显低于各自基准跑分（LightGBM 基线 IC 0.0308）。原因：标签改截面 z 分数、LSTM 改吃全 158 列、树模型在折内早停极早（最优迭代仅 4 轮）。组合仍挽回了头部排序——Rank IC 0.0252 约为 LightGBM 基线（0.0135）的 1.9 倍，使回测贴近基线。
4. **严格性的价值在设计层**：扩展窗保证元模型见到的折外预测不含未来信息；若用随机 KFold，折外指标会被未来信息抬高，裁判配比失真（跑分上不可见，但避免了假信心）。

方案 C（统一 z 分数）运行耗时约 25 分钟（3 折 × 3 基模型折外训练 + 全量重训 + 回测；NN 走 GPU 卡 0 约 3.3GB 空闲，LightGBM 走 CPU）。第一次运行因 `fold_ends` 最后一个日期写成了训练段终点（折外段为空）而失败，已修复并在模型类加了防御断言；失败 recorder 按惯例留档。实验记录：`examples/mlruns`（experiment 118604898218966411，C 为 recorder `75a191a7`、B 为 `ce0801ff`）；日志副本 `examples/nn_logs/stacking_2026-10-18.log`。

**方案 B 对照实验（同日追加）：各基模型用自己的基准卷子。** 起因是质疑"必须统一 z 分数"——原则上成立（线性元模型的权重可吸收量纲差异），遂改为三套基准原版口径（LightGBM 原始收益标签 / MLP z 分数且丢 VWAP0 / LSTM 20 列筛选 + 秩归一标签），裁判目标与折结构不变。结果（同口径）：

| 指标 | 方案 C（统一 z 分数） | **方案 B（各基准口径）** | 基线 LightGBM | 基线 LSTM |
|---|---|---|---|---|
| IC | 0.0232 | 0.0090 | 0.0308 | 0.0029 |
| Rank IC | 0.0252 | **0.0490** | 0.0135 | 0.0438 |
| 含成本年化超额 | **+11.08%** | **-6.67%** | +11.35% | -11.08% |
| 信息比率（含成本） | 0.864 | -0.525 | 0.940 | -0.849 |
| 最大回撤（含成本） | -16.24% | -20.27% | -15.55% | -25.00% |
| 元模型权重 | 55/33/12 | lgb 0.478 / mlp 0 / lstm 0.164 | — | — |

方案 B 的教学要点（比 C 更有料的一次失败）：

1. **Rank IC 全项目最高档，回测却最差之一**：0.049 的秩 IC 接近 HIST（0.0576），含成本超额 −6.67%。全截面排序好、头部排序差，TopkDropout 只吃头部——"IC 高 ≠ 赚钱"（3.3 节 Linear 教训）的最强案例。
2. **混合量纲下权重不可读**：LightGBM 折外权重 0.478、LSTM 0.164，名义配比约 3:1；但实测两列预测标准差 0.022 vs 0.066（取自基准批次 pred.pkl），实际影响约 1:1——裁判"说"的配比与裁判"做"的配比差 3 倍。原则上权重可吸收量纲，实践上权重失去解释性。
3. **皮尔逊线性裁判消化不了"秩强皮尔逊弱"的成分**：名次分口径的 LSTM 折外 Rank IC 0.0448（全列最高）、皮尔逊 IC 仅 0.008；按 MSE 拟合的线性裁判给它约一半实际影响力，测试段头部排序崩坏拖垮整体。结论：要么统一目标（方案 C），要么元模型拟合前把各列折外预测按列标准化（标准做法，未测，记为 B'）。
4. **非负约束的自动淘汰生效**：MLP 折外 IC −0.005（负贡献），权重恰为 0——裁判不需要人工干预就把有害成分清出门外。

方案 B 运行备注：LSTM 批大小恢复官方默认 800（本机基准跑分的 200 是共享 GPU 显存紧张时的权宜值）、DataLoader worker 降至 8（当日服务器外部 CPU 负载 215）；其余超参与各自基准一致。日志副本 `examples/nn_logs/stacking_2026-10-18_B.log`。



> 本章把两批共 13 项实验的结论收拢成一张证据网，并用机器学习理论解释每个关键概念"为什么如此设计"。读者对象：已跑通示例、想理解背后原理的研究者。

### 6.1 实验结论总表

**日频系列**（数据窗口：特征 2020-01-02 起；除特别注明外训练 2020~2023、验证 2024、测试 2025-01-01~2026-07-01；标的全集 csi300；收益均为相对沪深 300 的年化超额）：

| 实验 | 模型/方法 | IC | Rank IC | 含成本年化超额 | 信息比率 | 最大回撤 |
|---|---|---|---|---|---|---|
| workflow_by_code | LightGBM+Alpha158 | 0.0287 | 0.0141 | +10.90% | 0.890 | -13.46% |
| 基线 LightGBM | 同上（yaml 超参） | 0.0308 | 0.0135 | +11.35% | 0.940 | -15.55% |
| 基线 CatBoost | CatBoost+Alpha158 | 0.0393 | 0.0138 | **+14.66%** | **1.132** | -16.04% |
| 基线 XGBoost | XGBoost+Alpha158 | **0.0421** | 0.0150 | +12.41% | 0.999 | **-14.35%** |
| 基线 DoubleEnsemble | LightGBM 集成筛选 | 0.0303 | 0.0081 | +8.31% | 0.646 | -18.52% |
| 基线 Linear | 线性回归 | 0.0082 | **0.0501** | -8.37% | -0.662 | -24.79% |
| 基线 Stacking（5.11） | LightGBM+MLP+LSTM 堆叠，非负线性元模型 | 0.0232 | 0.0252 | +11.08% | 0.864 | -16.24% |
| 基线 Stacking-B（5.11，各基准口径） | 同上，基模型各用各的标签口径 | 0.0090 | **0.0490** | -6.67% | -0.525 | -20.27% |
| portfolio 指数增强 | LightGBM+风险模型优化 | 0.0308 | 0.0135 | -5.14%（不含成本 +0.35%） | -0.497 | -17.26% |
| 超参优化 (Alpha158) | Optuna×20 trial | —— | —— | 最优验证 L2 0.997258（trial 16），优于官方基线超参的 0.9984~1.0009 | | |
| 超参优化 (Alpha360) | Optuna×20 trial | —— | —— | 最优验证 L2 0.997816（trial 10） | | |

**分钟系列**（1min 特征 2024-01-02 起）：

| 实验 | 说明 | 关键结果 |
|---|---|---|
| highfreq 数据层 | 360 万行分钟特征加载/序列化/窗口重配 | 全部成功；Handler 缓存使同类工作流从 31.6s 降至 2.5s（日频对照） |
| highfreq 1min 建模 | LightGBM 二分类（预测下一分钟涨跌），测试段 2024-07 下半月 | 分钟 IC 0.0222、Rank IC 0.0285、多头精确率 51.6%（随机基准 50%） |
| 滚动训练数据层 | 3 轮年度滚动（train 3 年+valid/test 各 1 年） | 每轮三段数据就绪；"取数一次、逐轮重加工"模式验证 |
| 内存复用演示 | 同一工作流重复执行，第二遍直接复用进程内 Handler 对象 | 基线 119.9s（两次平均口径）→ 复用后 52.4s / 48.2s（约 2.3 倍加速） |
| 嵌套执行回测 | 日/5分/1分三层 + SBB/TWAP 算法交易，21 个交易日（2024-03） | 成功（exit 0）；嵌套机制全程打通，日级指标 FFR 为 NaN（内层逐笔信息聚合的已知怪癖）；首跑两个月窗口曾在第 30 日 0% CPU 挂起，缩至 4 周后稳定完成 |

### 6.2 十二条核心结论（实验证据 × 理论解释）

**结论 1：树集成显著优于线性模型，因为量价因子与收益的关系是非线性的。**
证据：同数据同切分下，CatBoost（IC 0.0393、含成本年化超额 +14.66%）、XGBoost（IC 0.0421、+12.41%）与 LightGBM（0.0308、+11.35%）全面领先线性回归（IC 0.0082、-8.37%）；三棵不同的"树"彼此差距远小于它们与线性模型的差距，说明提升来自模型族而非具体实现。
理论：线性模型是高偏差模型，当真实映射 f(x) 非线性时系统性欠拟合；GBDT（梯度提升决策树）通过**加性集成**逐轮拟合当前残差的负梯度——每棵浅树是低方差弱学习器，串行提升主要降低偏差，等价于函数空间的梯度下降。表格类（tabular）中低维稠密特征上，树集成至今仍是强基线，这正是"先跑 LightGBM 再谈深度模型"的行业惯例的来源。

**结论 2：IC≈0.03 在 A 股日频上已是"有信息"的信号，因为金融预测的信噪比天然极低。**
证据：本次所有日频模型 IC 落在 0.01~0.04 区间，且能转化出 8%~12% 的年化超额（含成本）。
理论：IC 是预测分数与次日真实收益的截面皮尔逊相关。市场接近有效时可解释方差极小——若单日截面 R² 达 1% 就已非常显著（R²≈IC²，0.03 的 IC 对应约 0.09% 的方差解释率）。与图像识别（准确率>95%）不同，量化建模是在噪声里提取微弱、但被大样本放大的信号：1600 个交易日 × 每日几百只股票的截面，使微弱相关足以统计显著并转化为组合收益。

**结论 3：评估指标必须与决策规则对齐——Linear 的 Rank IC 全场最高，组合却亏钱。**
证据：Linear 的 Rank IC 0.0501 为四模型之最，但 TopkDropout 组合含成本年化 -8.37%。
理论：斯皮尔曼秩相关度量**整个截面**的单调相关性，而 TopkDropout 只消费**分数最头部**（前 50 名）的排序质量——全截面中度单调与尾部精确排序是两回事；线性模型的预测是特征的加权和，头部排序容易被少数极端特征值带偏。机器学习的一般教训：离线指标（proxy metric）与最终决策效用（task metric）脱节时，应构造与决策一致的评估（例如直接评估 top-k 命中率或分组收益），qlib 的 SigAnaRecord + 回测双层评估正是这个用意。

**结论 4：信号只是半成品，组合构建层同样决定成败。**
证据：**同一份 LightGBM 信号**（IC 同为 0.0308），TopkDropout 含成本年化超额 +11.35%，而默认配置的指数增强策略 -5.14%。
理论：从预测到收益要经过"预测→权重→成交"两跳。TopkDropout 的权重由排序硬约束（每日只换 5 只）而天然低换手；指数增强示例未设换手上限，优化器每日完全重排权重，把微弱信号（不含成本仅 +0.35%）在费用上亏穿。这对应 ML 系统设计里的"预测器-决策器解耦"：决策器引入的偏差（约束形式、成本模型、再平衡频率）可以放大或吞噬预测器的全部边际。

**结论 5：交易成本是年化 5 个百分点量级的"隐形标签偏移"。**
证据：三个盈利基线的不含成本→含成本年化超额均下降约 4.8~5.5 个百分点；指数增强从 +0.35% 到 -5.14%。
理论：成本 ≈ 换手率 × 双边费率 × 频率。训练目标（未来收益）不包含成本项，相当于优化的是"无摩擦世界"的收益——若把成本作为标签或约束纳入训练/决策（如预测净收益、在优化里加换手惩罚），模型才会学会"少动"。这是从学术 benchmark 到实盘必须补上的一层。

**结论 6：金融数据必须按时间切分，禁止随机 K 折交叉验证。**
证据：本报告全部实验统一使用时间顺序的 train（2020~2023）/valid（2024）/test（2025~2026.07）三段；两批官方示例同样按时间切分。
理论：金融时序存在强自相关与分布漂移，随机打乱会造成**信息泄漏**（用未来训练预测过去）与**前视偏差**（lookahead bias，特征或统计量包含了决策时不可知的信息）。qlib 的 DatasetH.segments 即时间切分的配置化实现；滚动训练（结论 10）是它在非平稳世界里的推广。

**结论 7：截面标准化是在剔除"市场共同因子"，让模型学相对强弱而非绝对涨跌。**
证据：所有 Alpha158 工作流的 learn_processors 都含 CSZScoreNorm/CSRankNorm（标签按日截面 z 分/秩标准化）；日志中可见处理器逐项生效。
理论：某日全市场普涨 3% 时，个股收益的绝对值主要反映"今天的市场"而非"个股的好坏"。按日做截面标准化等价于减去当日截面均值（一因子模型里市场因子的代理），把标签变成"相对当日同侪的超额表现"；秩版本（CSRankNorm）对重尾分布更稳健（金融收益厚尾，极端值会主导 z 分数的尺度）。这与 ML 里"去除 batch 共模干扰、保留判别性方向"的思想一致。

**结论 8：早停是 GBDT 的隐式正则化。**
证据：每个 LightGBM 实验日志均出现"Early stopping, best iteration is: [n]"——验证损失连续 50 轮不改善即停止，最优轮数常远小于上限。
理论：GBDT 的有效容量随迭代轮数增长，轮数越多训练误差单调下降而验证误差先降后升（过拟合）。早停在验证误差最低点冻结模型，等价于控制假设空间复杂度；配合学习率（每棵树的步长）构成"收缩+早停"的双重正则。LGBModel 把 evals_result 暴露给上层（如超参优化的目标函数），就是把这一泛化信号用作外部决策。

**结论 9：超参数优化本身也会过拟合——验证集是有限资源。**
证据：Optuna 20 次试验把验证 L2 从基线约 0.9984 压到 0.997258（约 0.1% 的相对改进），且搜索空间很宽（学习率 0~1 均匀采样）。
理论：每次"依验证集表现选超参"都在消耗验证集的统计效力，试验次数多了会逐渐拟合验证集噪声（超参过拟合），因此严谨做法是嵌套验证或多验证集确认。TPE（树形 Parzen 估计器）是序贯基于模型的优化（SMBO）：用已完成试验拟合"好/坏参数"的两个密度模型，从更可能好的区域采样，比网格/随机搜索在高维空间高效得多——但搜索效率越高，越要警惕对验证集的过度挖掘。本实验目标函数取**验证集**损失、测试段从未参与调参，正是这条纪律的执行。

**结论 10：滚动训练是对概念漂移的标准应对。**
证据：rolling_process_data 完成 3 轮年度滚动；每轮的处理器（标准化统计量）只用当轮训练段重新拟合（IT_FIT_SEQ 模式）。
理论：金融时间序列非平稳——风格因子效力随政策、资金结构漂移（概念漂移 concept drift）。一次训练用数年存在"模型老化"；滚动重训让模型追踪最近的市场状态。更关键的细节是处理器统计量必须随窗口重拟合：若用全历史均值标准化，等于让早期数据的统计量"泄露"进后期窗口，属于隐蔽的前视偏差。qlib 把"取数一次缓存（省算力）"与"加工逐轮重做（防泄漏）"分离成两层，兼顾效率与正确性。

**结论 11：频率越高、单 bar 信噪比越低——样本量与单样本信息量的权衡。**
证据：1min 二分类的 IC 0.0222、多头精确率 51.6%（随机基准 50%），单分钟多空收益仅 0.54bps；而日频 IC 0.0287 却支撑起 10%+ 年化超额。
理论：分钟 bar 主要由微观结构噪声（买卖价差跳动、流动性冲击、订单流不平衡）构成，可预测成分占比更低；收益的增加靠"次数 × 单次微弱优势"累积。高频建模因此更依赖：①更大样本量（360 万行 vs 日频 29 万行）压低估计方差；②特征工程贴近订单簿动力；③执行层（TWAP/SBB 拆单）把纸面优势转化为成交。这正是 NDE 嵌套执行示例存在的意义——信号与执行在高频下不可分。

**结论 12：结构化协方差是用低秩结构正则化一个"维数灾难"的估计问题。**
证据：portfolio 第一步对全市场约 5800 只股票、240 个交易日收益估计协方差——观测数 T 远小于维度 N。
理论：样本协方差在 N>>T 时是病态的（秩最多 T-1、特征值谱严重偏离），直接求逆做组合优化会放大极端权重。结构化估计 cov(X)=F·cov(B)·Fᵀ+diag(var(U)) 把协方差约束为"低秩公共因子部分 + 对角特定风险"：PCA 取前 10 个主成分作 F，相当于把有效参数量从 O(N²) 降到 O(N·k)，是典型的**谱正则化/降维**思想。这一框架也是 Barra 类商业风险模型的统计学基础。

### 6.3 关键概念速查（理论定义 → 在本报告中的出现位置）

| 概念 | 理论定义 | 出现位置 |
|---|---|---|
| IC / Rank IC | 预测与真实收益的截面皮尔逊/斯皮尔曼相关（按日计算取均值） | 全部 SigAna 结果 |
| ICIR | IC 的均值÷IC 标准差——信号的"夏普比率" | 同上 |
| 前视偏差（lookahead bias） | 使用决策时点之后才可知信息造成的虚假优势 | 结论 6、10 |
| 截面标准化（CSZScoreNorm/CSRankNorm） | 按日截面去均值除以标准差（或转秩），剔除市场共同模式 | 结论 7 |
| 早停（early stopping） | 验证损失不再下降即停止训练的正则化手段 | 结论 8 |
| 梯度提升（GBDT） | 逐轮拟合残差负梯度的加性树集成，主要降偏差 | 结论 1 |
| TPE / SMBO | 用概率模型引导下一步采样的序贯超参优化 | 结论 9 |
| 概念漂移（concept drift） | 输入-输出联合分布随时间变化 | 结论 10 |
| 信噪比（SNR） | 可预测方差占总方差比例；IC²≈截面可解释方差 | 结论 2、11 |
| 结构化协方差估计 | 低秩因子部分+对角特定风险的协方差正则化 | 结论 12 |
| 实现差额（implementation shortfall） | 纸面收益与可成交收益之差（费用、冲击、滑点） | 结论 4、5 |
| TopkDropout | 每日持分数前 k 名、剔除最差 n 名的排序型组合规则 | 全部回测 |






## 7. 第三批：服务型示例（MongoDB / Arctic / 在线服务 / 强化学习，2026-10-05）

> 环境前提见 [环境_MongoDB与Arctic.md](环境_MongoDB与Arctic.md)。本批四组各跑通了代表性流程。

### 7.1 model_rolling/ —— TaskManager 滚动训练

- 流程：`reset → task_generating（RollingGen 按步长 550 个交易日滑动生成任务）→ task_training（TrainerR 本地训练）→ task_collecting（RecorderCollector + RollingGroup 聚合）`，一条 `python task_manager_rolling.py main` 跑通（exit 0）。
- 结果：滚动 2 轮 × 2 模型（XGBoost/LightGBM）共 4 次训练，聚合出的 LGBModel 预测覆盖测试段 2022-07-01 ~ 2024-10-09（csi300）。
- 适配：mongo 地址改 127.0.0.1；任务日期重设（train 2020~2021、valid 2022H1、test 2022H2 起滚）；**csi100 成分股数据只更新到 2020-10-20**（过期的标的池直接导致"空数据"），改用 csi300。
- 教学点：TaskManager 把"任务"物化为 MongoDB 里的文档（可跨进程/跨机领取），TrainerR/TrainerRM 分别对应本地与远程训练模式；滚动聚合时 RollingGroup 按"模型×测试段"拼接预测。

### 7.2 online_srv/ —— 在线服务模拟（online_management_simulate）

- 流程：OnlineManager+RollingStrategy 模拟在线滚动更新：窗口内按 rolling_step=80 天触发任务重训并产出最新信号，最后统一收集信号并回测。
- 结果（模拟窗口 2024-07-01 ~ 2024-09-30，信号实际覆盖 2024-06-03~09-30，TopkDropout top30/drop3，相对沪深 300）：**含成本年化超额 +3.76%（信息比率 0.250、最大回撤 -8.69%）；不含成本 +8.52%**（exit 0）。
- 适配：mongo 地址；任务模板日期重设（train 2024Q1/valid Q2/test 6 月）；csi100→csi300；回测基准 SH000903（中证 100）本机无数据改 SH000300。
- 教学点：在线模拟把"研究训练"与"信号产出"解耦成定时任务流，是 qlib 面向生产的一侧；同样体现出"换手成本"对短窗口信号的侵蚀（8.5%→3.8%）。

### 7.3 orderbook_data/ —— Arctic 订单簿存储与查询

- 两步全通：`create_dataset.py`（arctic 环境）把 2020-12-31 单日的 Tick/Transaction/Order 三类数据（约 4000 只股票、tar.gz 解压后逐股入 ChunkStore，joblib 并行）导入 MongoDB；`example.py`（qlib 环境，pytest）5 个查询测试全部通过——tick 级 `$ask1/$bid1` 盘口、`TResample($ask1,'1min')` 重采样、transaction/order 频率查询（样例 SZ000725，09:15 起真实盘口）。
- 适配：查询侧 provider_uri 指向本机分钟库 cn_data_1min；qlib 环境补装 arctic（`--no-deps` + pymongo 3.11 + lz4 + tzlocal——qlib 的 ArcticFeatureProvider 自身 import arctic，而 pip 全量装会拉旧 pandas 编译失败）；顶部的 arctic 导入加保护（查询侧其实不需要）。
- 教学点：非定频数据（逐笔/盘口）没有共享日历，qlib 用 Arctic 后端 + `align_time=False` + `time2idx=False` 的表达式路径支持；`freq="ticks"/"order"/"transaction"` 直接查 ChunkStore。

### 7.4 rl/ —— 强化学习订单执行（simple_example notebook）

- `jupyter nbconvert --execute` 全单元执行成功（0 报错），玩具环境的 PPO 训练输出正常（如 `Action = 60.000004, Reward = 0.600000`）。
- 依赖矩阵（关键坑）：qlib.rl 面向**旧 gym + tianshou 0.4.x**——pyproject 钉 `tianshou<=0.4.10`。实测：tianshou 2.0（模块重构，No module named 'tianshou.policy'）、0.5.1（要 shimmy 桥接且 reset 返回格式断言失败）均不可用，**装 0.4.10 一次通过**。numpy 2.4 下 gym 0.26 的兼容性担忧未成真。
- 未跑的兄弟示例：`rl_order_execution/`（完整订单执行 RL）需先用其脚本从回测收集订单数据再长时训练，耗时数小时；其依赖栈已由本 notebook 验证齐备。


### 5.8.1 受阻三模型的攻关记录（2026-10-07，三缺三补）

**HIST（成功）**：官方 `stock2concept/stock_index` npy 按其 2008~2020 宇宙构建（735 股 × 4635 概念的 0/1 矩阵 + 股票→行号字典），与自建宇宙不匹配直接 IndexError。解法：写 `build_local_concept_npy.py` 从本机概念成分文件（`instruments/concept_tzt_*.txt`，390 个概念）按本机 csi300 宇宙（811 只历史成员）重建同构 npy（每股平均 7 个概念、163 只零概念），yaml 指向本地文件后一次跑通（GPU）。结果：IC 0.0119、**Rank IC 0.0576（全部 22 个模型中最高）**、含成本年化超额 -12.62%——又一次"秩相关好、组合亏钱"的时序模型样本。

**ADARNN（成功）**：复盘确认是"共享 GPU + 并发波动"误伤的小模型。关键发现：此前 `CUDA_VISIBLE_DEVICES=1` 屏蔽后进程只见一块卡，而 yaml 里 `GPU: 6` → 设备序号越界（`invalid device ordinal`）。**CUDA_VISIBLE_DEVICES 屏蔽下进程内序号必须写 GPU: 0**。修正后干净单跑一次通过：IC 0.0080、Rank IC 0.0511、含成本年化超额 -7.03%。

**TFT（成功，共十九次尝试，指标已完整落库）**：TF1 代码迁到 TF 2.21 CPU 的完整踩坑链，每层修复都是一类通用兼容知识：
1. `tensorflow-cpu` 安装（GPU 版会与 vLLM 抢显存）；
2. **并行装包风险**：装 TF 中途升级 protobuf，导致同时启动的 HIST/ADARNN 导入崩溃（共享环境的教训：装包时别并行跑任务）；
3. 会话 API 垫片：Keras3/TF2.21 移除了 `keras.backend.get/set_session`，在 tft.py 顶部按旧语义补齐（get=当前默认会话， set=进入 as_default）；
4. **符号回填**：libs/* 以裸 `import tensorflow` 调 `variable_scope/Saver/ConfigProto` 等 TF1 符号——把 compat.v1 的缺失属性批量 setattr 回顶层 tensorflow 模块；
5. numpy 2 移除 `np.Inf` → np.inf；
6. 硬编码 `use_gpu=(True,...)` 触发 CUDA 加载失败（CPU 版 TF）→ 改 False；
7. `KerasTensor.get_shape` → `tf.TensorShape(x.shape).as_list()`（Keras 3 的 .shape 可能是元组）；
8. **KerasTensor 不能喂原生 tf 算子**（Keras 3 结构性限制）→ `TF_USE_LEGACY_KERAS=1` 走 tf-keras（Keras 2）路径；
9. `sample_weight_mode="temporal"` 旧训练路径需 `optimizers.legacy.Adam`（含 get_updates）；
10. tf-keras LSTM 梯度 while 循环 bug（`while_grad: Connecting to invalid output 4`）→ LSTM `unroll=True`（implementation=1 无效）；
11. 数据侧：本流程**从不调用 transform_inputs**，month(1~12)/year(20xx)/const(1.0) 原始值直接进嵌入层必然越界（12 月=索引 12 不在 [0,12)）→ 类别零基化（month-1、year/const 置 0）；另 identifier 词表按训练段拟合 → 验证/测试过滤为训练段标的集合。
最终结果：**IC 0.0238、Rank IC -0.0369、含成本年化超额 +12.08%（NN 族第一）、信息比率 0.645、最大回撤 -19.40%**——一个绝妙的对照教材：其皮尔逊 IC 在 NN 中仅次于 MLP、回测赚钱，而秩 IC 为负，与时序模型们"Rank IC 高却亏钱"的病症恰好互为镜像（TopkDropout 实际消费的是分数与收益的线性关系和头部集中度）。第 12~19 次尝试补齐的兼容项：12) 保存/加载用 `tf.train.Saver`（compat.v1.train 子模块回填）；13) 运行时 `tf.keras.backend` 在 legacy 模式下解析为 `tf_keras.api._v2.keras.backend`——与常见导入路径都**不是同一模块对象**，垫片必须无条件覆盖含运行时对象在内的全部 backend（部分还带返回 None 的桩）；14) keras fit 会弹出默认会话，会话垫片须有状态且进出栈；15) `reset_default_graph` 在嵌套图下断言失败而实际非必需（各阶段显式新建 Graph），注释掉；16) qrun 收尾把模型存 params.pkl，TF 图不可 pickle——TFTModel 补 `__getstate__` 剔除图对象（权重已在检查点）。指标在最后一步崩溃前已全部落库（实验状态 FINISHED），该崩溃为纯装饰性。

至此全部 18 个基准模型在本机跑通。


### 6.4 Alpha158 与 Alpha360 怎么选（通俗版）

先把两者想象成做菜：**Alpha158 是加工好的半成品**——人已经把量价数据算成 158 个现成指标（动量、波动率、量价相关性……），拿来就能用；**Alpha360 是原材料**——只是把过去 60 个交易日、每天 6 个数（开高低收、均价、成交量）摆成一张 60×6 的表格，唯一的处理是每个数都除以"今天"的值（这样 5 块的股票和 500 块的股票可以放在一起比），**它不是因子，是一段走势的原始记录**。

**什么时候用 Alpha360？**

1. **模型本身会"看走势"的时候。** LSTM、Transformer 这类时序模型的能力恰恰是读序列的形状——先后、远近、周期。给它们 158 个压缩好的指标，等于把一张 K 线图缩成一行摘要，特长发挥不出来；给 360 的原始矩阵，它们才有东西可"看"。官方示例也是这么安排的：8 个时序模型只配了 Alpha360、没有 Alpha158 版本。
2. **想验证"模型能不能自己学出规律"的时候。** 用 158 等于人把答案提示写在卷面上；用 360 才是真正的闭卷考——模型能从原始走势里学出多少东西，一试便知。做因子研究时，它也是天然的"零加工对照组"。
3. **想调整"回看多远"的时候。** 窗口 60 天可以随意改长改短，做敏感性实验很方便。

**什么时候别用？**

快速出结果、或者用树模型（LightGBM/XGBoost）的时候。树模型不会看形状，只会一列列比数值——给它 360 个生数据，不如给它 158 个加工指标。我们自己的调参实验就是证据：同一套搜索、同一个 LightGBM，**Alpha158 的最优解是"364 叶的深树 + 很小的学习率"，Alpha360 的最优解却是"13 叶的浅树 + 较大的学习率"**。直白地说：生数据噪声大，模型必须"记性差一点"才不容易把噪声背下来（浅树＝记性差＝不容易过拟合）。

整批实测也印证这个分工：Alpha158 配树模型，含成本年化超额 +11%~+15%；Alpha360 配的 8 个时序模型多数反而亏 5%~16%（只有 TFT 例外，+12.08%）。**一句话总结：白天干活用 158+GBDT，研究时序模型和表示能力时才上 360，并且要准备好跟噪声多斗几个回合。**
