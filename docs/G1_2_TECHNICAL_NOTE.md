# G1.2：节点自适应空间门控 ST-FedNet

项目名称：**基于时空注意力机制与联邦学习的配电网负荷预测**。

## 计算结构

新增独立工厂名称 `stfednet_gated`，对应 `GatedSTFedNet` 和 `SpatialGatedFusion`。既有 G1/G1.1 模型、参数键名及形状保留。

```text
X                          [B,48,N,1]，使用训练集参数标准化
G = GAT(X,edge_index)      [B,48,N,32]，双头，使用原有物理图与自环
S = Linear(32,1)(G)        [B,48,N,1]
gate = sigmoid(Linear(2,1)(concat([X,S],dim=-1)))
                           [B,48,N,1]
X_fused = X + gate*S       [B,48,N,1]
H = GRU(X_fused)           [B,N,48,32]，原单层共享GRU
C = AdditiveAttention(H)  [B,N,32]，原单头加性时间注意力
Y_pred = Linear(Dropout(C))[B,1,N]
```

门控仅有一个共享 `Linear(2,1)`，权重 `[1,2]` 初始化为零，偏置 `[1]` 初始化为 **−3.0**；总计三个门控参数。初始 `sigmoid(-3)=0.047425873`，所有样本、历史步和节点起始门控相同；训练后依据各自 X、S 独立计算，可随样本、时间、节点变化。门控范围为 [0,1]，不依赖节点编号，正常参与反向传播。恒等残差直接保留 X。没有新增 GAT 层、Transformer、门控 MLP、额外时序分支、特征或数据集。

原有 GRU/时间注意力类、损失、标准化、有效负荷掩码和四项指标复用。`forward` 仍只返回预测张量；`forward_with_gate` 在同一次前向中提供预测及门控，供验证统计使用，无需额外重复 GAT/GRU 前向。全部参数在 39、50、56、76、80 节点客户端上名称与形状一致，可用于后续模型聚合。

## 协议与公平比较

`configs/g1.yaml` 未修改：2021 年、48→1、按目标时间 70/15/15、Hidden32、Heads2、Dropout0.1、Batch32、LR0.001、AdamW、Huber、最多30轮、Patience8、Seed42、CUDA优先。标准化仅拟合训练时段，损失和四项指标排除无负荷节点，MAE/RMSE 的单位为 kW、WAPE/MAPE 为百分比。MAPE 仍使用 |真实负荷|≥0.1kW 的阈值和有效占比。

每轮验证 WAPE 选优，训练不读取测试窗口。新运行位于 `results/g1_2/stfednet_gated/<client>/seed42_<时间>/`，保存 Best、Latest、配置、train.log、epochs.csv 和最佳门控统计 JSON。旧运行结果不被覆盖。

G1.2 是本轮最后一次指定结构改进，开发结束后原则上冻结预测网络，不因验证结果反复扩展结构。用户完成 seed42 验证对照后，若有优势，再决定 seed2026/3407 与真实图/仅自环拓扑消融；门控大小不能单独证明物理关联有效。

## 门控监控范围

`GateMean` 和 `GateStd` 从验证集的同一次预测前向取得，model处于eval模式，统计 **本轮全部已评估窗口 × 48历史步 × 全部节点**，包括中间无负荷母线。门控统计的节点范围与预测损失掩码不同，因中间母线也参加物理图计算。

正式训练默认无 batch 限制，覆盖每客户端全部 1314 个验证窗口。重叠窗口中的历史观测按窗口出现次数计入统计。采用float64按元素数合并批次均值和中心二阶矩，标准差为总体标准差（ddof=0），最后一个小batch不会被赋予过高权重。没有用最后一个训练batch替代验证统计。

每轮终端和train.log增加 GateMean/GateStd，epochs.csv增加 gate_mean/gate_std 列。Best/Latest含 `validation_gate_statistics`，并在每次验证WAPE选优时生成 `best_gate_statistics.json`，其中记录epoch、seed、最优WAPE、均值、标准差、元素计数、验证窗口数、总窗口数、统计定义及覆盖是否完整。此文件对应Best epoch，而非最后训练epoch。

短冒烟仅使用前4个验证窗口（batch2×2），明确标记 `complete_validation=false` 和 `smoke=true`。统计不能作为完整验证或正式空间贡献结论。

## PyCharm正式训练命令

工作目录 `E:\Dachuang_FedSTA`，解释器 `E:\Dachuang_FedSTA\.venv\Scripts\python.exe`。无需重建环境或新增依赖。

五客户端顺序训练（只训练和验证）：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_clients --model stfednet_gated --device cuda --output-root results/g1_2
```

默认读取完整 G1 YAML，seed42和所有科学配置保持一致。批次摘要写入 `results/g1_2/batch_summary.json`，包括各客户端Best指标、epoch、路径和Best门控统计。单客户端及中断恢复：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --model stfednet_gated --client lv_39 --device cuda
.\.venv\Scripts\python.exe -m scripts.train_model --resume "results/g1_2/stfednet_gated/lv_39/<运行目录>/latest.pt" --epochs 30
```

恢复读取已保存的模型、优化器、早停及随机状态，从最近完整epoch继续。门控日志随新epoch追加，Best门控统计仅在WAPE改善时更新。

全部正式训练完成后，用四模型验证Checkpoint比较：

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_g12 --g1-root results/g1 --g11-root results/g1_1 --g12-root results/g1_2 --seed 42
```

比较脚本只加载checkpoint中的验证元数据，不加载数据或运行测试。输出 `results/g1_2/validation_comparison.json`，包含每客户端四模型的MAE/RMSE/WAPE/MAPE、Best epoch、checkpoint路径、Gated门控统计、五客户端等权平均WAPE和Gated优于各对照的客户端数。默认要求五客户端齐全；可用--clients调试子集，输出会标明是否完整五客户端。

脚本沿用G1.1比较规则，选择各模型/客户端/seed下修改时间最新的正式Best，而非跨重跑挑选最低WAPE。跳过Smoke checkpoint；校验客户端、seed（元数据与配置）、48→1、数据签名及原协议中的科学配置（隐藏维度、dropout、batch、lr、epochs、patience、优化器相关参数、Huber阈值、MAPE阈值等）。结果带绝对路径便于核对。G1.1原比较入口保留。

## 测试及CUDA冒烟

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke_g12 --device cuda --output-root results/g1_2_smoke
```

本机使用Python3.13.7、PyTorch2.10.0+cu126、RTX4060LaptopGPU。完整回归 **83项通过**，包含G0/G1/G1.1原测试及新增五节点数前反向、门控初始化/范围/梯度更新、恒等残差、节点参数一致、样本/时刻/节点自适应、门控统计逐元素对照、Best/Latest严格加载和精确恢复、Best门控JSON与CSV、禁止访问测试窗口及四模型元数据比较检查。

五客户端CUDA短冒烟全部通过：每轮2个训练batch和2个验证batch、从epoch1恢复到epoch2、门控权重/偏置与GAT投影更新、预测和门控维度/范围正确、Best验证指标与统计重载一致。每客户端测试窗口均替换为禁止访问对象；本次未访问。未运行正式长训练或正式测试集，未开始G2。

开发前Git源码与新代码中的 **八种旧模型** 在相同seed下初始化参数和预测逐元素一致。本机G1/G1.1的 **48个现有Best/Latest checkpoint** 均严格加载，无旧文件改写。

证据见 `results/g1_2/validation.json`、`results/g1_2/compatibility.json` 和 `results/g1_2_smoke/summary.json`；完整短运行终端记录及CSV保存在冒烟目录。以下真实lv_39日志只反映工程验证，不代表正式性能：

```text
Epoch 01/1 | TrainLoss=0.356073 | ValMAE=0.8915 | ValRMSE=1.0978 | ValWAPE=76.7551 | ValMAPE=110.1375 | Best=76.7551 | Time=0.50s | GateMean=0.047371 | GateStd=0.000061
Epoch 02/2 | TrainLoss=0.344528 | ValMAE=0.8956 | ValRMSE=1.1015 | ValWAPE=77.1096 | ValMAPE=110.8486 | Best=76.7551 | Time=0.03s | GateMean=0.047407 | GateStd=0.000124
```

G1.2已具备用户在PyCharm开展seed42正式验证对照的工程条件。正式Gated比较结果需用户训练后生成，本次不宣称性能改善。源码、文档及紧凑验证日志提交main，权重留本地；已有G1/G1.1正式结果与PyCharm暂存文件保留。
