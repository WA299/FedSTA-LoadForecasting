# G3.1 窗口级 DP-FedAvg 开发说明

## 范围与结论

G3.1 新增独立的客户端窗口级 DP-SGD 与服务器 FedAvg 流程，支持 `stfednet_gated` 和 `gru_attention`。它复用现有冻结模型、`PreparedClient`、训练集归一化、验证指标及FedAvg状态聚合；旧G0/G1/G2/G2.1/G2.2训练路径没有改动。正式20轮DP训练和测试集评估均未运行。

当前实现适用于数值与工程研究，**没有启用可作为正式隐私声明依据的安全噪声源，`privacy_claim`会在读取数据前拒绝启动**。本地RDP值是建立在理想独立高斯噪声假设上的理论会计结果，不是当前普通PyTorch PRNG训练获得的隐私保证。训练损失、梯度诊断、验证指标、验证模型选择均是非DP输出，应限制访问，不应直接公开。

## 隐私单位、邻接和边界

一条记录是一张固定图的一个完整滑动窗口：48小时全节点输入，加上1小时全节点目标。一个图中的不同母线、时间步或输入/目标元素不作为独立记录。使用固定拓扑、load mask、训练集拟合Scaler和其他预处理参数条件下的窗口级 add/remove 邻接。数据集中的记录数 `N=6084`，期望Batch `B=32`，每一步抽样率 `q=B/N=0.0052596976`；名义本地Epoch取 `ceil(N/B)=191` 步，20轮共3820步/客户端。五客户端每轮全部参与，Accountant没有使用客户端参与抽样放大。

窗口有48小时输入，且相邻窗口高度重叠。移除一个窗口不会移除其他窗口中重复出现的底层小时，所以窗口级保证不能解释成原始小时、家庭、用户或客户端级保证。同一底层曲线如果跨客户端重复，某个来源变化可能影响多个客户端的输出，需要按该来源的贡献跨客户端组合隐私损失；当前每客户端单独输出的ε没有进行这种跨客户端用户级组合。拓扑、负荷掩码、Scaler、数据签名以外的训练预处理、验证集选择、非DP验证统计也没有被这套窗口级训练机制保护。特别是训练集数据拟合的Scaler属于私有派生量，报告中的邻接是条件化于Scaler固定，不能推出端到端原始数据隐私。

## 更新和采样实现

每客户端每个DP更新独立对该客户端的全部训练窗口进行Bernoulli(q)抽样，允许真实Batch大小小于、大于期望值，亦允许空Batch。没有打乱后切成固定Batch。对实际入选的窗口分别计算有效负荷节点上的 masked Huber mean，逐窗口求所有可训练参数的完整梯度，并在全参数拼接范数上裁剪到 `C=1.0`。只有完成逐记录裁剪后才对梯度求和。

对每个可训练参数张量，向裁剪梯度和加独立高斯噪声，标准差为 `σC`；随后用固定 `qN=32` 除，而不是实际随机Batch大小。空Batch仍生成同尺度噪声、写入零梯度并执行AdamW更新，同时累计一个RDP步骤。没有使用普通Batch梯度裁剪。AdamW优化器每个客户端通信轮次重置，在该轮的所有本地DP步之间保留优化器状态；FedAvg按唯一训练窗口数加权聚合五个客户端模型状态，采样序列、拓扑、节点IDs、load mask和Scaler不进入参数聚合。

`src/privacy/per_sample.py`保留显式逐窗口循环作为数值Oracle。CUDA候选为 `torch.func.functional_call + grad_and_value + vmap`，关闭cuDNN的GRU实现；它在两模型的真实裁剪状态下通过逐参数裁剪聚合结果对照，因此本机CUDA `auto`选择该路径。测试条件下的数值比较设Dropout=0以比较相同函数值；Dropout=0.1时两个路线各自独立执行并确认梯度有限，随机Dropout掩码不同，不将其逐元素输出误称为相同。`torch.func`在GRU仍会触发PyTorch慢速fallback警告。遇到不支持算子时直接抛错，不会静默切换到循环；用户可显式指定 `--gradient-method loop`。

## 会计器与复现

每客户端独立使用已安装的 Opacus 1.6.0 `RDPAccountant`；未使用Opacus `GradSampleModule`或`DPOptimizer`。Accounting只在一次带噪AdamW步骤完成后增加，checkpoint会同时保存Accountant history与实际step count。恢复检查训练进度、每客户端步数与Accountant状态是否一致；客户端本地每一个提交步骤都保存模型、AdamW状态、Poisson采样器与随机状态。这样恢复时从下一步继续，不会重复已提交步骤。通信轮次只有在五个客户端都完成并聚合、验证后才作为完整轮次提交。

配置是 `C=1`、`δ=1e-5`、`q=32/6084`、191步/本地Epoch、20轮。以下只是在该Poisson抽样与独立理想高斯噪声假设下，用同一版本RDP会计器计算的**理论示例**，尚未执行这些正式训练：

| 噪声乘数σ | 每客户端理论ε | δ | 假设总步数 |
|---:|---:|---:|---:|
| 0.5 | 15.6347 | 1e-5 | 3820 |
| 1.0 | 1.9793 | 1e-5 | 3820 |
| 2.0 | 0.6936 | 1e-5 | 3820 |
| 4.0 | 0.3078 | 1e-5 | 3820 |

这些不是训练结果、也不能用于当前research模式作出实际隐私声明。σ=0控制组仍记录DP采样和裁剪步骤，但噪声为零，Accountant输出ε为`null`，并明确标记非DP。既有G2 FedAvg使用固定规模随机洗牌Batch，与这里Poisson采样协议不同，应作为外部非DP基线而不能套用本会计器。普通Seed让开发检查可复现，但会把噪声状态保存在research checkpoint中；它不适用于安全部署。G3.0检查的Windows/Python3.13环境中Opacus `secure_mode=True`因`torchcsprng`不可用而失败。G3.1没有编写自有密码学随机数实现，也不将依赖包存在性当成安全证明。正式`privacy_claim`必须等经过审查的安全Gaussian RNG、种子及状态隔离完成后另行实现。

RDP组合只按一个客户端的实际Poisson更新次数累计。五客户端全参与并不增加同一客户端本地ε的客户端抽样放大；如果对手可见所有客户端更新，仍需按目标保护单位定义跨客户端组合。DP-SGD也不提供安全聚合、加密传输或客户端身份隐藏。

## 运行、恢复和输出

使用项目的 `.venv`。新结果写入有时间戳、模型和σ区分的 `results/g3_1/dp_fedavg/<model>/`子目录，保留已有G2结果。每个运行包含 `best_dp_global.pt`、`latest_dp_global.pt`、`config.json`、`rounds.csv`、`round_diagnostics.json`、`train.log`及最佳验证结果JSON。Checkpoint写明随机源、梯度方法、模型、协议、客户端、数据签名、每客户端RDP Accountant、进度与随机状态。

研究模式命令如下，所有验证按五客户端等权WAPE选优。命令均由用户在PyCharm终端执行；本次Codex只完成单步冒烟：

```powershell
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model stfednet_gated --noise-multiplier 0.5 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model stfednet_gated --noise-multiplier 1.0 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model stfednet_gated --noise-multiplier 2.0 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model stfednet_gated --noise-multiplier 4.0 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model gru_attention --noise-multiplier 0.5 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model gru_attention --noise-multiplier 1.0 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model gru_attention --noise-multiplier 2.0 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model gru_attention --noise-multiplier 4.0 --privacy-mode research --gradient-method auto --device cuda
```

可选σ=0控制组（非DP）命令：

```powershell
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model stfednet_gated --noise-multiplier 0 --privacy-mode research --gradient-method auto --device cuda
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --model gru_attention --noise-multiplier 0 --privacy-mode research --gradient-method auto --device cuda
```

从 `latest_dp_global.pt` 恢复：

```powershell
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.train_dp_federated --resume '<run目录>\latest_dp_global.pt'
```

每个σ组合由运行目录时间戳隔离，不覆盖其他实验。训练/梯度诊断和验证/选优均属于限制访问的非DP产物；不能将`rounds.csv`、best验证JSON或`train.log`公开并声称受到梯度DP保护。

## 验证结果

运行环境：Python 3.13.7、PyTorch 2.10.0+cu126、CUDA 12.6、Opacus 1.6.0、NVIDIA RTX 4060 Laptop GPU。G3.1新增14项单元测试；全仓152项测试通过。CUDA数值测试中，两模型在实际裁剪状态下`torch.func`与循环路线逐参数一致；Dropout梯度有限。旧G0至G3.0测试未修改且全通过。

五张真实图的短冒烟对两个模型各运行1轮、每客户端1次Poisson更新、验证最多1个Batch。节点数为39/50/56/80/76，固定`qN=32`；实际抽样Batch在本次随机实现中为26到42，所有客户端均严格写入每轮1次Accountant更新，Best/Latest严格加载通过。ST-FedNet Gated五客户端一轮与验证约2.64秒，GPU峰值分配约1616MB；GRU+Attention约1.90秒、约752MB。按单步计时线性乘191得到的本地Epoch粗估范围约43–99秒（Gated）和约32–57秒（GRU+Attention）；该单步时长包含模型初始化与原子检查点写入，Poisson Batch计算量又随随机Batch而变，因此只是粗略成本界限，不是实测Epoch或时限保证。G3.1短冒烟实际`torch.func`梯度堆栈占用约466–780KB（Gated）及466–753KB（GRU+Attention），并按客户端记录在机器报告中；该数字仅计逐样本参数梯度堆栈，不包含激活、优化器与CUDA缓存。未发生OOM，报告保存在 `results/g3_1/cuda_smoke_summary.json`，所有冒烟产物均标记`smoke=true`、`test_evaluated=false`及`privacy_claim_allowed=false`。

本阶段没有执行完整20轮训练。数值一致、工程恢复和冒烟通过仅证明代码路径可运行，不代表模型预测性能或正式差分隐私声明已获验证。
