# G1.1：ST-FedNet 空间残差融合

## 目标与边界

G1 验证集对照中，GRU + Temporal Attention 在五个客户端的四个上取得最低 WAPE；GAT 空间模块尚未带来稳定增益。G1.1 为其新增轻量模型 `stfednet_residual`，让模型从原始负荷起步，只通过小幅可学习残差利用图信息。原有 `stfednet`、`gru_attention` 及其他 G1 模型实现保持不变。

本阶段包括代码、单元测试及短时 CUDA 训练/验证冒烟。五客户端完整 30 epoch 训练由用户在 PyCharm 执行。G1.1 开发不运行独立测试评估，也不以短冒烟误差下结论。

## 模型定义

对输入 `X ∈ R^[B×48×N×1]`，沿用双头 Graph Attention 和已有邻接自环实现：

```text
G = GAT(X, edge_index)            # [B, 48, N, 32]
S = Linear(32 → 1)(G)             # [B, 48, N, 1]
X_fused = X + α · S               # [B, 48, N, 1]
H = GRU(X_fused)                  # 共享单层 GRU，逐节点编码48小时
C = AdditiveTemporalAttention(H)  # 单头，softmax沿时间维
Y_hat = Linear(C)                 # [B, 1, N]
```

`α` 是标量 `nn.Parameter`，初始化为 **0.05**，训练时与 GAT、GRU 等共同由 AdamW 更新。恒等残差保证 `α=0` 时 `X_fused` 精确等于原始输入；`α` 接近零时也保留直接的负荷信号通路。空间 GAT、投影、GRU 和输出层的参数形状均与节点数无关，可在五客户端间聚合。新模型的 state-dict 命名位于独立 `ResidualSTFedNet` 类，不改变原 `NeuralForecast` 的旧键名。

本实现的 `α` 为直接学习的实数，不限定为百分比或 0–1 范围，可学习正负修正。由于空间投影的尺度也能变化，仅观察 `α` 大小不能证明空间关联有贡献；需结合验证误差及未来拓扑对照解读。

每个 epoch 的终端和 `train.log` 增加 `SpaceScale=...`，`epochs.csv` 增加 `space_scale` 列，Best/Latest checkpoint 附加可读 `space_scale` 数值，并同时在 `model_state_dict` 保存权重。旧模型仍采用原有 CSV 列和 checkpoint 参数结构。

## 训练配置及客户端比较

沿用 `configs/g1.yaml` 不变：48→1、Hidden Dim=32、Batch Size=32、Learning Rate=0.001、Epochs=30、Seed=42、Early Stopping Patience=8；时间划分、AdamW、Huber Loss 和验证 WAPE 选优均不变。新 checkpoint 和日志放在 `results/g1_1/stfednet_residual/<client>/...`，避免覆盖既有 `results/g1/` 实验。

完整顺序训练五个客户端：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_clients --model stfednet_residual --output-root results/g1_1
```

脚本为各客户端运行同一 YAML 配置，并在 `results/g1_1/batch_summary.json` 记录 Best epoch、四项验证指标、可学习系数和 checkpoint 路径。它不会调用独立评估入口，也不会访问测试窗口。

新模型完成训练后，自动发现 G1 与 G1.1 中符合 seed 和数据划分配置的正式 Best checkpoint，对每客户端比较 GRU + Temporal Attention、原始 ST-FedNet 和残差 ST-FedNet：

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_validation --g1-root results/g1 --g11-root results/g1_1 --seed 42
```

结果写入 `results/g1_1/validation_comparison.json`，含每客户端验证 WAPE、Best epoch、运行目录、新模型胜出的客户端数及等权平均验证 WAPE。只加载 checkpoint 元数据中的验证指标，不载入数据、不访问测试集。多次重跑同 seed 时，脚本取每客户端、每模型最新的正式 Best checkpoint；会跳过标记为 smoke 的模型检查点，并拒绝数据签名、任务或训练配置不一致的结果。应先检查比较文件中客户端、Seed、Best epoch 与运行目录，再解读趋势。

如果验证集上的残差模型有优势，后续在预先约定的验证协议下增加随机种子；再比较真实物理拓扑与仅自环拓扑，检验收益是否来自实际空间关联。不要因单 seed 或反复查阅同一验证集结果夸大结论。任何正式测试集评价留待模型和选择方案冻结后独立进行。

单客户端完整运行仍可用：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --model stfednet_residual --client lv_39 --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 42 --patience 8
```

## 快速工程验证

`scripts.smoke_g11` 每客户端训练一轮后恢复到第二轮，每轮仅训练和验证各两个 batch，使用独立 `results/g1_1_smoke/` 路径。以下命令明确使用 CUDA：

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_g11 --device cuda --output-root results/g1_1_smoke
```

汇总 JSON 标记 `smoke=true` 和 `test_split_accessed=false`。冒烟数据不能用于正式模型选择或论文性能结论。

## 验收点

- 五种实际节点数（39、50、56、76、80）执行前向/反向传播，输出 `[B,1,N]`。
- GAT 投影输出与残差融合均为 `[B,48,N,1]`；置 `α=0` 时恒等路径精确保留输入。
- `α` 和 GAT 获得有效有限梯度并更新；五种 N 的 state-dict 参数名称和形状完全一致。
- 原始 `stfednet`、`gru_attention` 参数键不变，旧 checkpoint 可以严格加载。
- 新模型训练/恢复 checkpoint、终端和 CSV 均记录每轮 `SpaceScale`。
- 测试集在本阶段保持未使用；五客户端比较只读验证 WAPE。

## 本机验收结果（2026-10-13）

PyCharm 原解释器 `E:\Dachuang_FedSTA\.venv\Scripts\python.exe`，Python 3.13.7、PyTorch 2.10.0+cu126、RTX 4060 Laptop GPU。没有新增依赖或改动 `configs/g1.yaml`。

- 完整自动化测试通过，包含 G0/G1 回归及新模型五节点数反向、恒等残差、参数更新、参数跨客户端一致、checkpoint恢复、系数日志、验证结果比较与批量配置一致性检查。最终数量见 `results/g1_1/validation.json`。
- CUDA 冒烟五客户端全部通过，含 Best/Latest 保存与加载、从 epoch1 恢复到 epoch2、空间系数/GAT 投影参数更新以及有限四项验证指标。测试窗口被防访问对象替换，任何读取均抛异常；此次未访问。
- 旧 G1 源码取自开发前 Git HEAD，和新代码的七个旧模型在相同 seed 下参数及输出逐元素一致；本机 38 个既有 Best/Latest checkpoint 全部严格加载，未改写旧文件。证据为 `results/g1_1/compatibility.json`。
- 源码、配置、文档、紧凑 JSON/CSV/文本日志纳入版本管理；权重留在本地。既有 G1 用户结果目录和 PyCharm 暂存文件保持原样。

真实短冒烟示例（`lv_39`）：

```text
Epoch 01/1 | TrainLoss=0.392979 | ValMAE=0.8345 | ValRMSE=1.0365 | ValWAPE=71.8473 | ValMAPE=101.9169 | Best=71.8473 | Time=0.66s | SpaceScale=0.048778
Epoch 02/2 | TrainLoss=0.382497 | ValMAE=0.8377 | ValRMSE=1.0390 | ValWAPE=72.1193 | ValMAPE=102.5161 | Best=71.8473 | Time=0.02s | SpaceScale=0.049772
```

各客户端末轮系数如下；仅用于确认可学习参数会更新，不代表正式性能或空间贡献。

| 客户端 | 节点数 | 初始 α | 冒烟末轮 α |
|---|---:|---:|---:|
| lv_39 | 39 | 0.05 | 0.049772 |
| lv_50 | 50 | 0.05 | 0.046829 |
| lv_56 | 56 | 0.05 | 0.046692 |
| lv_80 | 80 | 0.05 | 0.053752 |
| industrial_mvlv | 76 | 0.05 | 0.053998 |

G1.1 已具备进行完整验证集比较的工程条件。还不能据本次短冒烟认定新增模型优于基线；正式性能等待用户在 PyCharm 运行上述五客户端命令后确认。
