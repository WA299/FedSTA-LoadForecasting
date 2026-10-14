# G2.2 FedPer 五客户端个性化联邦负荷预测

## 范围与算法

G2.2 在既有五客户端仿真上增加 FedPer，不修改 G1/G1.1/G1.2 预测网络，也不改 FedAvg、FedProx、Centralized 或 Local-only。允许的模型是 `stfednet_gated` 和 `gru_attention`。数据、拓扑、训练时间划分、训练集拟合 Scaler、load mask、kW 反归一化、48→1 任务及 G2 优化超参数保持一致。

FedPer 将标准 model state_dict 严格拆分：

- **客户端私有参数：**只有 `head.weight` 和 `head.bias`。
- **共享骨干：**其他全部模型状态，包括 Gated 的 GAT、空间投影和节点自适应门控，以及 GRU 和 Additive Temporal Attention。

五个 head 在同一确定性初始化之后从同一份初始预测头复制。每个客户端在每轮开始时接收相同的服务器骨干，并恢复自己上一轮的 head；客户端用完整模型和独立 AdamW 优化器训练一轮，然后只将骨干交给聚合器。聚合器执行按训练窗口数量加权的 FedAvg。客户端的新 head 留在对应 client ID 项下，跨轮继续使用，不能参加聚合或重新初始化。训练窗口数为聚合权重；本地多 epoch 时仍按唯一训练窗口数量计权，与 G2 FedAvg 一致。

这里的“服务器保存五个 head”仅指单机仿真为了检查点和恢复而集中序列化。真实个性化部署中，各客户端可以只保留自己的 head；FedPer 的更新仍可能泄露信息，本阶段没有实现 DP-SGD，也不提供严格差分隐私保证。

## 验证、诊断与检查点

聚合后，服务器对每个验证集分别装配 `SharedBackbone + LocalHead(client_id)`。指标是反归一化后的 MAE、RMSE、WAPE、MAPE，忽略无负荷母线；MAPE 遵循 0.1 kW 阈值。Best 由五客户端等权平均验证 WAPE 选择；pooled Overall WAPE 单独记录。Best 中的共享骨干与五个 head 必须来自同一通信轮次。

每轮保存客户端 Huber/Total loss、零 Prox loss、裁剪前梯度范数和裁剪比例，并生成 `round_diagnostics.json`。Latest 保存完整五客户端 head、最新骨干、最佳轮次的骨干和全部 head、轮次历史、数据签名、样本数、配置和 Python/NumPy/CPU/CUDA RNG 状态。恢复只从最近完成的整轮继续；每轮客户端 AdamW 按 G2 策略重新创建，不保存优化器动量。Best/Latest 仅含权重、元数据和随机状态，不含原始负荷或 Scaler 数组。

输出目录为 `results/g2_2/fedper/<model>/seed42_<timestamp>/`，含 `best_fedper.pt`、`latest_fedper.pt`、`config.json`、`rounds.csv`、`round_diagnostics.json`、`train.log` 和 `best_clients_validation.json`。冒烟运行使用独立临时目录，均带 `smoke=true`，比较器不会将其作为正式结果。

## PyCharm 命令

正式训练：

`E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.train_fedper --model stfednet_gated --device cuda`

`E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.train_fedper --model gru_attention --device cuda`

恢复命令：

`E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.train_fedper --resume <run目录>\latest_fedper.pt --device cuda`

正式默认协议是 `configs/g2_2.yaml`：Seed 42、20 轮、每轮一个本地 Epoch、Batch 32、AdamW、学习率 0.001、权重衰减 0.0001、Huber、隐藏维度 32、Dropout 0.1、五客户端全参与。FedPer 不使用 FedProx，配置里的 `mu=0.0` 只是满足共用 G2 配置校验，不进入训练目标。不要在 Codex 冒烟配置上续训成正式运行。

完成用户正式运行后生成已有方法验证汇总：

`E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.compare_fedper --seed 42 --rounds 20`

比较器读取 FedPer、FedAvg、FedProx 和 Centralized 的 Best/Latest 验证检查点，以及现有 Local-only 验证记录；不做前向推理、不读取测试集、不修改权重。FedProx 固定选用 G2.1 已完成验证敏感性实验中的 gated `mu=0.001`、GRU+Attention `mu=0.0001`，并在输出标注选择来源。重复运行按最近完成的 Latest 检查点时间和路径选择，不按最低分挑重复实验。输出 JSON 和 CSV，包含五客户端指标、等权/pooled WAPE、相对 FedAvg 和 Local-only 的差异、80 节点网及工业网变化、最佳轮次与训练预算。

## 开发验证

- 自动化回归：`python -m pytest -q`，**124 passed**；包括新增 8 项 FedPer 测试和全部 G0/G1/G1.1/G1.2/G2/G2.1 测试。
- CUDA 冒烟：RTX 4060 Laptop GPU 上，两个模型各用五个真实客户端运行 2 轮，每客户端每轮 1 个训练 Batch 和 1 个验证 Batch。检查了两个预测架构的共享参数可在 39、50、56、80、76 个节点上严格加载，head 更新及跨轮 client 映射，骨干专用加权聚合，Best/Latest 完整装载，以及连续/恢复权重和指标完全一致。产物在 `results/g2_2/cuda_smoke_summary.json`，checkpoint 临时文件在 `.pytest_tmp/`。
- 冒烟 WAPE 是部分 Batch 的工程检查值，绝非正式验证或测试性能；未运行 20 轮正式训练，测试 split 由守卫对象拒绝访问。
- G2/G2.1 Checkpoint 格式未修改；原模型构造和参数名称保持不变。当前本地 20 份 G2 Best/Latest FedAvg、FedProx、Centralized 检查点均已按原模型严格加载；核验摘要见 `results/g2_2/checkpoint_compatibility.json`。
- 开发时运行了 `scripts.compare_fedper`，汇总了现存验证记录。因尚无正式 FedPer Checkpoint，FedPer 条目明确标记为缺失；这不是 FedPer 结果。比较器的参考数值来自验证集，不可当作测试性能。单 Seed 结果不代表统计显著性，FedPer 也不保证优于 FedAvg、FedProx、Local-only 或 Centralized。

本阶段没有开发 FedAtt 或差分隐私，也没有执行正式联邦训练或测试集评估。
