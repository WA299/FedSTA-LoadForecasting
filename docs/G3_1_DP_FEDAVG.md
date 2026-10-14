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

每客户端独立使用已安装的 Opacus 1.6.0 `RDPAccountant`；未使用Opacus `GradSampleModule`或`DPOptimizer`。Accounting只在一次带噪AdamW步骤完成后、在内存中增加；RDP历史及实际step count只在写入Checkpoint时序列化，ε按会计器步数缓存，步数变化后才重算。默认`checkpoint_every_steps=25`，可通过`--checkpoint-every-steps 25`调整；显式设为1即每个本地步骤提交。客户端完成、五客户端聚合完成、轮次验证和Best选择之后仍强制保存。低于间隔的未提交步骤不会提前复制模型、AdamW、随机数或Accountant状态。

Checkpoint采用同目录唯一临时文件加原子替换；Windows替换短暂占用时有限重试。**最新成功原子替换的`latest_dp_global.pt`是唯一恢复提交点**。例如运行到第37步、最近提交为25步，中断恢复时模型、AdamW、Poisson采样器、噪声生成器、Dropout随机状态及RDP会计均回滚至25步，再重算26步起。客户端完成、聚合完成等提交点同样完整保存；聚合后、验证提交前恢复会重做确定性的聚合与验证，不会重复任何客户端DP步骤。最佳/最新权重及对应验证数据仍绑定同一完整轮次。Checkpoint强制点可能早于25步间隔发生，这是有意的耐久性边界。

每步仍有一次内存中的Accountant更新，但完整Accountant状态读取、ε查询和`privacy_accounting.json`写入移到Checkpoint。`src/privacy/accountant.py`只在步骤计数改变时重算ε。各阶段耗时汇总至受限的`checkpoint_performance.json`；精确CUDA阶段测量通过单独的profile pass完成并同步CUDA，不把profile额外同步计入常规墙钟比较。训练公式、Poisson抽样、噪声、归一化、AdamW和FedAvg均保持原样。

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

## Checkpoint开销优化验收

对照程序：`python -m scripts.benchmark_dp_checkpoints --steps 30 --profile-steps 2`。脚本从基线Commit `3bd0589e02490a30957c997ef9154effe87e00df`动态加载旧训练代码，在隔离模块运行；不检出旧分支、不修改现有文件。使用五个真实客户端、CUDA、Seed=42、σ=1、每客户端30步、单通信轮、相同Poisson采样预算和验证最多一个Batch。另对当前优化版间隔1进行对照，以证明恢复逐步保存后模型输出仍与旧提交相同。验证值仅作实现一致性检查，不是正式模型结果。机器汇总在`results/g3_1_checkpoint_optimization.json`。

| 模型 | 旧提交逐步保存 | 优化版间隔1 | 优化版间隔25 | 旧→间隔25 | Latest/全部Checkpoint写入：旧→间隔1→间隔25 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| `stfednet_gated` | 37.02 s | 21.34 s | 19.08 s | 1.94× | 157/158 → 158/159 → 13/14 | 1735.3 MiB，各模式相同 |
| `gru_attention` | 33.92 s | 19.09 s | 16.90 s | 2.01× | 157/158 → 158/159 → 13/14 | 802.4 MiB，各模式相同 |

两种模型在旧提交、优化版间隔1和间隔25之间的全局参数、Best参数、每客户端Accountant状态、最终RNG状态和最佳验证结果均逐项完全一致。优化版间隔1比旧提交多一次Latest写入，这是按要求新增的聚合后强制提交点。Profile pass另外计时Poisson/Batch构造、逐窗口梯度与裁剪、AdamW、模型/优化器及RNG复制、Checkpoint写入、Accountant状态读取、ε查询和JSON写入。两步profile中，baseline→优化版间隔25的ε查询累计时间为Gated 1.696→0.037秒、GRU 1.719→0.037秒；状态复制分别为0.038→0.005秒、0.023→0.004秒；Checkpoint写入为0.076→0.032秒和0.062→0.027秒。逐窗口梯度仍是主要计算成本；两步profile样本较小，不据此得出梯度速度结论。Poisson/Batch和AdamW耗时变化很小。用户先前记录的Gated 56.57秒、GRU 34.16秒与本次旧版单次实测37.02/33.92秒存在环境波动，以上加速比不构成固定时限保证。

自动化恢复测试在research模式使用固定随机状态。间隔1与间隔25得到完全相同的参数、Accountant、RNG和验证结果；模拟第37步完成后异常退出时，磁盘Checkpoint仍严格记录第25步。恢复后重算未提交步骤，与连续40步的五客户端FedAvg输出一致，已提交Accountant步骤未重复记录。所有冒烟均不访问测试集，结果标记`smoke=true`及`privacy_claim_allowed=false`。本优化没有改动G3.1机制，也没有运行正式20轮训练。

对两种模型各挑选一组已有旧版正式目录，校验旧配置（没有`checkpoint_every_steps`字段）能恢复读取，Best/Latest均严格装载，已完成Run恢复不执行额外DP更新，并确认源Checkpoint文件SHA-256未改变、RDP步数/状态未变化。全仓自动化测试本轮为159 passed；新增检查覆盖间隔1旧版一致、间隔25保存频次、37步异常回滚、CUDA两模型未提交步骤恢复、聚合后恢复、Accountant ε缓存以及原子替换失败时旧提交保留。CUDA torch.func原有GRU batching-rule警告仍会出现，不影响测试结果。
