# G2 基础联邦负荷预测框架

## 范围

G2 实现了五客户端的单机联邦仿真和集中式对照，固定使用 `stfednet_gated` 与 `gru_attention` 两种已冻结的 G1/G1.2 预测网络。包含 FedAvg、FedProx、Local-only 结果汇总接口和 Centralized；默认每轮五客户端全参与；不包含 FedPer、FedAtt、DP-SGD 或正式测试集评估。

五个客户端为 `lv_39`、`lv_50`、`lv_56`、`lv_80` 和 `industrial_mvlv`。每个客户端继续保留自己的负荷矩阵、物理 `edge_index`、节点编号、有效负荷掩码和训练集拟合的 StandardScaler。训练窗口仍为过去 48 小时预测未来 1 小时，目标时间按 70%/15%/15% 顺序切分。训练和验证路径只接触 `PreparedClient.train` 与 `PreparedClient.val`，测试窗口由独立评估入口隔离。

## 模块和协议

- `src/fl/client.py` 创建一个临时本地模型，载入服务器轮次开始时的全局状态，按客户端图逐 Batch 训练，并返回模型状态、客户端训练窗口数和平均本地损失。返回值不含原始数据、拓扑、节点 ID、mask 或 scaler。
- `src/fl/aggregation.py` 对浮点模型状态按 `n_k / sum(n_j)` 加权平均；模型没有节点数相关参数。FedProx 的附加项是 `mu/2 * sum((w_local-w_global_start)^2)`，只遍历可训练参数，`mu=0` 直接退化为 FedAvg 本地 Huber 目标。
- `src/fl/server.py` 每轮向五个客户端发送相同状态，收集更新并聚合，随后用同一个全局模型逐客户端计算物理 kW 下的 MAE、RMSE、WAPE、MAPE。最佳轮次由五客户端等权平均验证 WAPE 选择，同时记录合并有效观测后的 overall WAPE。
- `src/fl/centralized.py` 使用一个共享模型和一个 AdamW 优化器，通过独立 DataLoader 按确定性的跨客户端 Round-Robin Batch 顺序交错训练；不同节点数不会被拼接到同一图张量。

每轮日志同时写入终端、`train.log` 和 `rounds.csv`。`best_global.pt` 与 `latest_global.pt` 保存完整的模型/轮次元数据、客户端列表、数据签名、配置、RNG 状态和验证摘要。恢复从最近完成的通信轮次的下一轮开始，不重复聚合已经完成的轮次。最佳客户端验证结果保存为 `best_clients_validation.json`。

客户端每轮重新初始化 AdamW，不保留跨轮本地动量。FedAvg/FedProx 使用相同初始化及由 `(seed, round, client_index)` 确定的 Batch 顺序和 Dropout 随机流。近端参考权重在本轮开始时冻结；轮间聚合权重使用唯一训练窗口数，不使用节点观测数量或本地 Epoch 的重复访问次数。正式五网均为 6,084 个训练窗口；冒烟对子集计数，每客户端为 4 个。日志 TrainLoss 是按实际处理窗口数加权的本地目标均值（FedProx 包含近端项）。Centralized 每轮为每个客户端保留独立 DataLoader，按 `lv_39 → lv_50 → lv_56 → lv_80 → industrial_mvlv` 的固定顺序轮流消费一个 Batch；当某客户端当前 local epoch 耗尽后从活跃队列移除，所有客户端耗尽后进入下一 local epoch。`local_epochs` 控制每个客户端完整迭代的次数；未设置 `max_train_batches` 时，每个 local epoch 完整使用五客户端各自的训练窗口；冒烟或受限运行中，则对每个客户端可见的窗口前缀完整分批消费。每个 local epoch 中，可见数据的所有 Batch 恰好处理一次。Batch 不跨图拼接，全程共享同一个模型和优化器。每轮的训练随机流由 Seed 和轮次确定，客户端 DataLoader 各自使用确定性的 shuffle generator；因此从最近完整轮次 checkpoint 恢复后，后续轮次与连续运行一致。

Latest 是完整轮次的原子提交点，其中还保存 `round_history`。恢复时由它修复 CSV 和 Best 派生文件，避免落盘过程中中断造成重复轮次行。文本日志保留运行尝试，CSV 是完整轮次曲线的依据。恢复只允许修改总轮数、设备、CPU 线程数和输出位置；训练协议和 Batch 上限必须一致，禁止将 smoke 恢复成正式实验。同一硬件、依赖和配置下验证了精确恢复；跨设备或不同数值库版本不承诺逐位一致。

## 配置和运行

默认配置位于 [`configs/g2.yaml`](../configs/g2.yaml)，正式运行由用户在 PyCharm 终端执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_federated --algorithm fedavg --model stfednet_gated
.\.venv\Scripts\python.exe -m scripts.train_federated --algorithm fedprox --model stfednet_gated --mu 0.01
.\.venv\Scripts\python.exe -m scripts.train_centralized --model stfednet_gated
.\.venv\Scripts\python.exe -m scripts.train_federated --algorithm fedavg --model gru_attention
.\.venv\Scripts\python.exe -m scripts.train_federated --algorithm fedprox --model gru_attention --mu 0.01
.\.venv\Scripts\python.exe -m scripts.train_centralized --model gru_attention
.\.venv\Scripts\python.exe -m scripts.compare_g2 --root results/g2
.\.venv\Scripts\python.exe -m scripts.compare_g2 --local-only --output results/g2/local_only_validation.json
```

恢复示例：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_federated --algorithm fedavg --model stfednet_gated --resume results/g2/fedavg/stfednet_gated/seed42_xxx/latest_global.pt
```

开发冒烟命令为：

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_g2 --device cuda
```

它只跑两轮、每客户端两个训练 Batch 和一个验证 Batch，覆盖两个模型、FedAvg、FedProx 以及两个 Centralized 控制，所有产物置于 `results/g2_smoke/` 并明确标记 `smoke=true`。冒烟不会读取测试 split，也不能作为正式性能结果。

## 指标解释和边界

每个客户端的指标只在有效负荷节点上、反归一化后的 kW 中计算。五客户端等权 WAPE 是模型选择指标；overall WAPE 将所有客户端的有效观测充分统计量合并后计算，二者不能互换。Local-only 是五个独立模型的结果汇总，不是一个共享的联邦模型，并在比较结果中保留各自训练预算。

比较入口只读现有权重内的验证摘要，排除 smoke，按最新完成的运行选择各方法/模型/Seed；不会在多个运行之间挑最低误差。它校验数据签名、Seed、48→1 任务、归一化划分及共享训练配置，并输出每网四项指标、MAPE 有效占比、等权/整体 WAPE 和客户端胜出方法，保存 JSON/CSV。FedProx 记录 `mu`；不同比例及本地 Epoch/轮次预算不混合。Local-only 的预算差异保留，不将其当作计算量完全匹配的结果。G1 Best 缺少充分统计量时，从已存验证 MAE/RMSE/WAPE/MAPE 和计数代数重建 pooled 指标，标注来源，未重新访问测试集。已生成 `results/g2/local_only_validation.json` 与 `.csv`，后续正式 G2 结果由用户训练后再汇总。

本阶段的联邦模块是实验仿真，不代表已经部署通信系统，也不保证参数更新没有信息泄露。尚未加入差分隐私，因此不报告隐私预算或 epsilon。后续 G3 必须先明确滑动窗口记录单位、相邻关系、裁剪、采样和隐私会计假设；窗口中的节点维度不自动构成独立 DP 记录。

## 验收记录

G2 新增单元测试覆盖聚合数值、样本数加权、FedProx 梯度、`mu=0` 等价性、变节点图、状态隔离、验证指标、checkpoint/恢复、Centralized Round-Robin 实际 Batch 顺序、比较器 smoke 排除和测试集禁用。本次 G2 调度修正新增真实 Batch 顺序与多 `local_epochs` 恢复检查；完整回归测试 `101 passed`。已有两种模型的十份正式本地 Best 均通过 strict 加载，未修改预测模型代码、G0 数据接口或旧权重。

CUDA 短冒烟在原有 G2 验收中覆盖 2 个模型 × FedAvg/FedProx 两轮和两个 Centralized 控制，每客户端每轮使用两个训练 Batch，输出 `smoke=true`，五个真实客户端均完成；本次修正再次运行同一五客户端 CUDA 冒烟，六种模型/算法组合、断点恢复与连续两轮权重一致性全部通过，输出位于 `.pytest_tmp/g2_round_robin_smoke/`。节点数为 39/50/56/80/76，使用 RTX 4060 Laptop GPU，Python 3.13.7、PyTorch 2.10.0+cu126、CUDA 12.6。原摘要为 `results/g2_smoke/summary.json`，原终端日志汇总为 `results/g2_smoke/smoke.log`；权重留在本地。测试窗口由守卫对象禁止访问。Codex 没有执行正式 20 轮联邦训练或正式测试集评估。
