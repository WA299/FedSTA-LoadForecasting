# G1 开发与验收报告

项目名称：**基于时空注意力机制与联邦学习的配电网负荷预测**。

## 交付结论

G1 工程开发完成，具备启动正式模型对照实验的条件，等待总控窗口验收。本阶段只运行单元测试与少量 batch 冒烟，未启动正式长时间训练，冒烟误差不代表正式模型性能。

## 实际运行环境

| 项目 | 实测值 |
|---|---|
| 项目根目录 | `E:\Dachuang_FedSTA` |
| PyCharm 解释器 | `E:\Dachuang_FedSTA\.venv\Scripts\python.exe` |
| Python | 3.13.7 |
| PyTorch | 2.10.0+cu126 |
| CUDA 运行时 | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 驱动 / 显存 | 560.94 / 8188 MiB |
| CUDA 前向及反向检查 | 通过 |
| pip 依赖检查 | 无冲突 |

正确的 `.venv` 已保留；误建的 `venv` 已按用户要求删除。环境记录在 `results/g1/environment.json`，依赖见 `requirements.txt` 和完整 `requirements-lock.txt`。图层使用原生 PyTorch，无 PyTorch Geometric 依赖。

## 工程文件

- 模型：`src/models/forecasting.py`。
- 训练、checkpoint、恢复及独立评估：`src/training/engine.py`；四项指标与客户端汇总：`src/training/metrics.py`。
- CLI：`scripts/train_model.py`、`evaluate_model.py`、`summarize_results.py`、`smoke_g1.py`、`check_environment.py`。
- G1 配置：`configs/g1.yaml`；测试：`tests/test_models.py`、`tests/test_training.py`、`tests/conftest.py`。
- 更新 `AGENTS.md`、README、依赖及 `.gitignore`；G0 loader 仅去除一个重复的同名 property 定义，保留原接口。
- 运行指南：`docs/G1_RUN_GUIDE.md`；真实日志和小规模结果：`results/g1_smoke/`、`results/g1_cli_smoke/`。

## 七种预测方法与共享模块

| 名称 | 空间模块 | 时间模块 |
|---|---|---|
| Persistence | 无 | 最近一小时观测 |
| Seasonal Naive | 无 | 目标时刻前 24 小时观测 |
| GRU | 无 | GRU 末状态 |
| GRU + Temporal Attention | 无 | GRU + Additive Attention |
| GCN + GRU | 物理图 GCN | GRU 末状态 |
| GAT + GRU | 双头 GAT | GRU 末状态 |
| ST-FedNet | 双头 GAT | GRU + Additive Attention |

ST-FedNet 为 GAT → GRU → Additive Temporal Attention → Linear。GAT 按实际物理拓扑加入自环，两个头拼接得到 32 维节点特征，历史各时间步共享参数。单层 GRU 输出全部 48 个历史状态，时间权重使用 `vᵀtanh(Wh+b)` 并沿时间 softmax。Dropout 0.1；所有参数独立于节点数量。三个消融方法复用相同的 GAT、GRU 和时间注意力类，无 Transformer 或额外门控。

当前小图使用稠密邻接掩码，复杂度随节点数平方增长，适用于本阶段 39–80 节点网络；未来更大网络可替换为空间稀疏实现。

## 统一数据、训练与评价协议

复用 G0 `load_grid` / `prepare_client`：2021 年 8760 小时，按目标时间 70%/15%/15% 划分，每客户端训练/验证/测试窗口为 6084/1314/1314。标准化仅拟合训练时段，验证和测试历史输入可以包含其预测时刻以前已观测的数据。

| 张量 | 形状 |
|---|---|
| 历史输入 X | `[B,48,N,1]` |
| 目标与预测 Y | `[B,1,N]` |
| edge_index | `[2,E]`，物理支路双向化 |
| load_mask | `[N]` |
| 目标 timestamps | `[窗口数,1]` |

五个 N 分别为 39、50、56、80、76；中间无负荷节点保留，仅在损失与指标中屏蔽。G0 的 48→24 配置能力与测试保留。

训练使用归一化有效节点上的 Huber Loss、AdamW、固定 seed42、patience8。每轮仅计算验证集 MAE/RMSE/WAPE/MAPE，以验证 WAPE 保存 Best，同时保存 Latest。checkpoint 包含模型、优化器、epoch、最优 WAPE、配置、seed、scaler、节点 IDs、数据签名；Latest 保留随机状态和早停计数。恢复从上一个完整 epoch 继续，未完成的 epoch 重跑。

正式测试使用独立入口加载 Best。四项指标先反归一化，MAE/RMSE 为 kW，WAPE/MAPE 为百分比；MAPE 仅统计 |真实负荷|≥0.1 kW 并报告有效占比。无有效分母时记为 null。结果保存真实值、预测值、时间、节点 IDs、mask。跨客户端汇总包含整体统计和客户端等权 WAPE，并拒绝模型/seed/协议混用。

G0 已记录的低压 MW→kW 为源文件缺乏单位字段情况下的明确假设，正式报告仍需依据数据源说明核实；工业小时 kWh 数值换算为平均 kW。工业网两个物理连通分量和跨客户端重复曲线保留，详见 G0 报告。

## 自动化测试与冒烟证据

使用正确 E 盘解释器执行 `python -m pytest -q`，结果 **47 passed in 18.04s**，包括已有 6 项 G0 数据测试。

- 7 个模型 × 5 种节点数：输出形状、可训练模型前向/反向、损失与有限值检查。
- 空间邻接约束、时间注意力归一化、Seasonal Naive 时刻偏移与物理单位指标检查。
- Best/Latest 序列化及加载、恢复后与连续训练权重完全一致的 CPU 测试、早停、训练不得访问测试窗口检查。
- 预测保存及日志内容检查，客户端等权 WAPE 与混合协议拒绝检查。

真实五网 CUDA 冒烟 **35/35 组合通过**，记录于 `results/g1_smoke/summary.json`。batch size2，每轮仅 2 个训练和验证 batch；25 个神经模型组合从 epoch1 恢复到 epoch2，10 个无参数基线各运行一次；全部 checkpoint 重新加载成功。各组合独立执行 2 batch 测试接口检查，结果有限。

训练、独立评估和五客户端汇总 CLI 均实测通过。`results/g1_smoke/stfednet_summary.json` 已包含五客户端整体统计和等权 WAPE，并标记 `smoke=true`。限 batch checkpoint 不能作为正式测试输入，需明确指定冒烟评估参数。

## 实际终端日志片段

来自 `results/g1_smoke/stfednet/lv_39/train.log`：

```text
Model=stfednet | Client=lv_39 | Device=cuda | BatchSize=2 | LearningRate=0.001 | Input=48 | Horizon=1 | Epochs=1 | Seed=42
Epoch 01/1 | TrainLoss=0.336930 | ValMAE=0.6628 | ValRMSE=0.8455 | ValWAPE=57.0647 | ValMAPE=78.5438 | Best=57.0647 | Time=0.02s
[Checkpoint] New best model saved
Epoch 02/2 | TrainLoss=0.306906 | ValMAE=0.6585 | ValRMSE=0.8378 | ValWAPE=56.6923 | ValMAPE=78.3753 | Best=56.6923 | Time=0.02s
[Checkpoint] New best model saved
```

上述时间为 GPU 初始化后的少量 batch 耗时，不能用于估计正式完整 epoch 耗时。完整训练日志同时保存在文本和 CSV 中。

## 下一阶段与版本管理

总控验收后，用户在 PyCharm 按运行指南开展七种方法的正式对照实验；先固定验证集选择方案，再独立测试，不按测试结果调参。G2 可复用不依赖节点数的模型工厂及训练/指标模块开展 FedAvg/FedProx。

G3 需明确窗口级 DP 记录与节点维度的区别：一个窗口包含全部节点和历史，节点不作为独立样本；重叠窗口不能直接宣称用户级隐私。隐私邻接关系、采样与会计器将在 G3 定义。

源代码、配置、文档、环境记录和紧凑日志/JSON/CSV 提交到既有 main；原始 Datasets 已在原 Git 历史中保留。模型权重与预测 NPZ 保留本地并忽略。既有用户暂存的 PyCharm `.idea` 文件保留，不纳入本阶段提交。最新 SHA 以交付汇报及 Git 日志为准。
