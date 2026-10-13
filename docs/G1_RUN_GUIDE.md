# G1 运行指南

项目：基于时空注意力机制与联邦学习的配电网负荷预测。

## PyCharm 环境

解释器选 `E:\Dachuang_FedSTA\.venv\Scripts\python.exe`，工作目录选 `E:\Dachuang_FedSTA`。PowerShell 终端可直接使用以下命令，无需依赖自动激活：

```powershell
cd E:\Dachuang_FedSTA
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest -q
```

已安装 Python 3.13.7、PyTorch 2.10.0+cu126。CUDA wheel 自带运行时，当前驱动 560.94 支持 CUDA 12.6；不需要另装 CUDA Toolkit。依赖版本记录在 `requirements.txt`，完整安装清单在 `requirements-lock.txt`，实测环境在 `results/g1/environment.json`。

重建环境时：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

没有 NVIDIA GPU 的机器将第一行换成 `--index-url https://download.pytorch.org/whl/cpu`。默认设备 `auto` 自动选择 CUDA 或 CPU，也可以显式传 `--device cpu`。

## 七种原始方法与 G1.1 新增模型

| 命令名称 | 方法 |
|---|---|
| `persistence` | 最近一小时观测值 |
| `seasonal_naive` | 目标时刻前 24 小时的观测值（48 小时窗口的第 25 个观测） |
| `gru` | 单层 GRU，末时刻隐藏状态 |
| `gru_attention` | 单层 GRU + Additive Temporal Attention |
| `gcn_gru` | 物理拓扑 GCN + 单层 GRU |
| `gat_gru` | 双头 GAT + 单层 GRU |
| `stfednet` | 双头 GAT + 单层 GRU + Additive Temporal Attention |
| `stfednet_residual` | GAT → 学习型小比例空间残差 → 同一 GRU + Temporal Attention |

可选客户端为 `lv_39`、`lv_50`、`lv_56`、`lv_80`、`industrial_mvlv`。模型复用相同 GAT、GRU 和时间注意力模块；所有可训练参数均不依赖节点数。

G1.1 的新增空间残差模型结构、批量命令和验证集比较见 [G1.1 技术说明](G1_1_TECHNICAL_NOTE.md)。

## 训练与恢复

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --model stfednet --client lv_39 --epochs 30
.\.venv\Scripts\python.exe -m scripts.train_model --model gru --client industrial_mvlv --epochs 30 --batch-size 32 --seed 42
```

配置为 `configs/g1.yaml`：2021 年、48→1、70/15/15 按目标时间切分、隐藏维度 32、Dropout 0.1、Huber Loss、AdamW、验证 WAPE 选优、patience 8。训练标准化只拟合训练时段；验证和测试输入可以引用其目标时刻以前的历史观测。五张网均有训练窗口 6,084 个、验证窗口 1,314 个、测试窗口 1,314 个。G0 默认 48→24 保留。

每次新训练创建独立目录 `results/g1/<model>/<client>/seed<seed>_<时间>/`，包含：

- `best.pt`：验证 WAPE 最优模型，含模型、优化器、epoch、最优 WAPE、配置、seed、标准化参数及数据哈希。
- `latest.pt`：最近完整 epoch，另含 Python/NumPy/PyTorch/CUDA/DataLoader 随机状态和早停计数。
- `train.log`、`epochs.csv`：与终端一致的日志和每轮指标。
- `config.json`：实际运行配置。

Persistence 和 Seasonal Naive 无训练参数，只作一轮相同划分的训练损失/验证指标计算，并保存可供独立测试的 checkpoint。

恢复命令使用本机实际路径替换占位符：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --resume "results/g1/stfednet/lv_39/<运行目录>/latest.pt" --epochs 50
```

恢复时读取保存的配置、模型和客户端，仅允许延长 epochs 或更改运行设备等执行选项。`Ctrl+C` 中断后，从最近完整 epoch 恢复；尚未完成的 epoch 会重跑。已触发早停的运行不会自动继续。每次恢复保留相关 `best.pt`。

## 独立正式测试

只有验证集选优和超参数确定以后才执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_model --checkpoint "results/g1/stfednet/lv_39/<运行目录>/best.pt"
```

训练过程中不会读取测试集。独立评估保存 `test/metrics.json`、`test/test.log`、`test/predictions.npz`。NPZ 包含 `y_true_kw`、`y_pred_kw`（均为 `[测试窗口数, 1, 节点数]`）、目标时间、原始母线 IDs 和 load mask；方便后续绘图。测试结果用于报告，不用于重新选超参数。

MAE/RMSE 单位 kW，WAPE/MAPE 单位 %，仅统计有效负荷母线；MAPE 限定 |真实值|≥0.1 kW，另记录有效占比。WAPE 分母为真实负荷绝对值总和。全零分母或无 MAPE 合格样本时输出 JSON `null`，不伪装成零误差。

跨客户端汇总同一个模型、同一个 seed 的五个 `metrics.json`：

```powershell
.\.venv\Scripts\python.exe -m scripts.summarize_results --metrics "<lv_39的metrics.json>" "<lv_50的metrics.json>" "<lv_56的metrics.json>" "<lv_80的metrics.json>" "<industrial的metrics.json>" --output results/g1/stfednet_summary_seed42.json
```

汇总同时计算全部观测的整体指标和客户端等权平均 WAPE，拒绝混合不同模型、seed、正式/冒烟结果或重复客户端。

## 冒烟检查

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --model stfednet --client lv_39 --epochs 1 --batch-size 2 --max-train-batches 2 --max-val-batches 2
.\.venv\Scripts\python.exe -m scripts.evaluate_model --checkpoint "<冒烟best.pt>" --max-batches 2
.\.venv\Scripts\python.exe -m scripts.smoke_g1 --output-root results/g1_smoke --device auto
```

全面 G1 冒烟每张网只加载一次，当前包括 G1.1 残差模型，共 8×5=40 个组合，每个神经模型训练 1 轮后恢复到第 2 轮，每轮仅 2 个训练和验证 batch。再次执行需指定新的空 `--output-root`。冒烟指标只用于工程检查，已写入 `smoke=true`，独立测试入口禁止将其当正式测试。G1.1 开发阶段只使用 `scripts.smoke_g11`，它只验证新增模型的训练/验证路径，不运行测试入口。

## 联邦与隐私扩展

G2 可复用模型工厂和训练/指标函数，参数形状在所有客户端一致。G3 必须区分一个训练窗口（包含所有节点和历史）与节点维度：窗口才是拟议的 DP 记录单位，节点不是独立记录；重叠窗口的隐私保证不等于用户级隐私。具体采样、邻接关系和会计器需在 G3 明确。
