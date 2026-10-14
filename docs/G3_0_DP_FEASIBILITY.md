# G3.0 差分隐私技术可行性报告

## 结论

G3.0 只验证逐样本梯度实现和环境兼容性，没有启动正式DP训练，也没有访问测试窗口。**目前已确认两种预测模型都能通过显式逐图窗口循环实现逐记录梯度、全参数裁剪及高斯噪声；CUDA `torch.func` 路径也在这台机器上通过数值对照。** Opacus 不能直接包装当前两种正式模型：原生GRU需要替换，GRU梯度样本轴会变成 `Batch × Node`，而Gated模型的自定义GAT还无法通过Opacus默认采样器处理。正式G3应先采用逐窗口循环作为参考实现，再评估CUDA `torch.func` 优化版，并保留循环路径做数值回归。

这只是工程可行性验证，不构成差分隐私保证或正式DP实验结果。

## 环境与代码变更

- Windows 11，Python 3.13.7。
- PyTorch 2.10.0+cu126，CUDA运行时12.6，NVIDIA RTX 4060 Laptop GPU。
- 新增并验证 Opacus 1.6.0；原PyTorch、CUDA和虚拟环境保持不变。`requirements.txt` 固定记录Opacus版本。
- 未修改 `src/models/forecasting.py`、G0数据接口、G1/G2训练代码、checkpoint格式或正式模型结构。
- 新增 `src/privacy/per_sample.py`，仅提供G3.0数值基准；新增环境检查、限量CUDA基准、测试与报告脚本。
- `results/g3_0/` 下的记录均标注可行性/冒烟用途。没有模型训练权重。

运行命令：

```powershell
E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.check_dp_compatibility --device cuda
E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.benchmark_per_sample_gradients --device cuda --batch-size 32
E:\Dachuang_FedSTA\.venv\Scripts\python.exe -m scripts.summarize_g30
```

脚本只从实际训练切分取一个限量batch。基准运行在数据加载后将测试Dataset替换为访问即报错对象；compatibility检查只用合成输入。报告字段显式记录 `test_split_accessed=false`。

## 隐私记录、邻接关系与适用范围

暂定一条训练记录为**一个完整的图时序训练窗口**：48小时全节点输入、1小时全节点目标，以及客户端固定图、load mask对应的计算过程。图中母线不是独立记录；一个batch内每个窗口各自计算损失和全模型梯度，再逐窗口裁剪。当前测试所定义的邻接关系为：固定客户端、模型、拓扑、load mask和标准化器后，训练样本集合增加或移除一条完整窗口（add/remove adjacency）。

这个定义只覆盖变换后窗口级记录，不等同于时间点、家庭、用户或客户端级保护：

- 滑窗每次向前移动一小时，任一内部原始小时通常会出现在最多48个输入窗口和1个目标窗口，即最多约49条训练记录中。修改一条原始小时数据可能改变多条窗口；窗口级DP不能直接解释成时间点级DP。任何更强的原始时间点结论都需要重新定义邻接，并分析组隐私和时间窗口暴露范围。
- 客户端之间如存在完全重复或高度相关的曲线，同一底层事件可能通过多个客户端更新被重复访问。各客户端分别的ε不能在该情形下直接视为底层事件的整体ε；应考虑跨客户端组合，保守组合会增大隐私损失。
- G0/G1按训练期原始负荷拟合每节点StandardScaler。该均值和标准差是私有数据派生量。若它们随私有输入改变，单纯对变换后窗口做DP-SGD不能覆盖整个预处理流程。正式G3应采用公开/固定参数，或单独进行有界、已计入预算的DP统计估计；不得发布未受保护的私有Scaler。
- 物理拓扑和有效负荷节点掩码作为模型输入/计算条件。若其本身属于私有信息，当前梯度保护并不自动隐藏它们；报告需明确以拓扑和mask公开或固定为条件，否则另行保护。
- 验证WAPE、逐客户端指标、按验证集选Best轮次或公开日志都不是训练DP自动保护的对象。正式发布前须使用公开评估数据，或对验证发布/模型选择另作隐私设计与记账。
- DP-SGD不提供安全聚合、传输加密、客户端身份隐藏，也不保护所有元数据。G2服务器仍可观察每个客户端上传的模型更新。

因此后续最多先声称“固定拓扑、mask和预处理条件下，训练窗口级DP”，且需经过正式实现、会计与随机源审计。不得将其扩展为端到端原始数据隐私或客户端级DP。

## 逐窗口损失、剪裁与噪声

设一个窗口的预测和目标为 `[H,N]`，客户端有效节点集合为 `M`。逐窗口损失定义为

\[
\ell_i = \frac{1}{H|M|}\sum_{h=1}^{H}\sum_{n\in M}
  \operatorname{Huber}(\hat y_{i,h,n},y_{i,h,n}).
\]

在客户端固定mask、每个窗口具有相同预测长度的前提下，`mean_i(ell_i)` 与现有 `masked_huber` 在整批 `[B,H,N]` 上的mean归约一致。逐窗口对全部可训练参数求梯度 `g_i`，而不是将节点分拆成记录。全参数L2范数后执行

\[
\bar g=\frac{1}{m}\left(\sum_{i=1}^{m}
  g_i\min(1,C/(\|g_i\|_2+10^{-12})) + z\right),
\quad z\sim\mathcal N(0,(\sigma C)^2I).
\]

该噪声尺度对应add/remove邻接下的裁剪梯度和；replace-one邻接的最坏敏感度是 `2C`，必须相应调整噪声定义/尺度。噪声加在**裁剪梯度总和**上，再除以归一化量。固定大小batch的末尾不完整batch使用其实际条数 `m` 归一化；Poisson采样则使用期望batch大小 `q*n` 归一化，即使实际batch较小或为空也保持这一分母，空batch仍进行噪声更新。DP-SGD会计必须和正式选定的采样过程、分母及敏感度定义一致。

`masked_huber_per_sample()`和`per_sample_clipped_gradients()`已单独实现上述测试逻辑。循环函数是可读、显存较低的正确性基准；它不接入G2客户端、没有训练循环、没有隐私会计或生产安全噪声源。

## 逐样本梯度路线检查

### Opacus 1.6.0

`ModuleValidator`报告两种模型的 `nn.GRU` 不受默认GradSample hook支持，建议替换成Opacus `DPGRU`。验证中，将原生GRU权重通过非严格方式载入转换模型后，转换前后前向输出在当前合成样例上数值相等；但转换后的state_dict出现原模型没有的GRU子层键，现有正式checkpoint不能直接strict-load，故没有修改正式模型。原生GRU的直接GradSample backward也因梯度样本广播形状失败。

Opacus默认路线仍不满足本项目的完整图窗口记录定义：

- `gru_attention`转换模型的GRU输入把 `[B,N]`展平成 `B×N` 条独立序列。GRU参数的 `grad_sample`首维因此为 `B×N`，而时间注意力和head的首维是 `B`。实际 `DPOptimizer.step()`因grad_sample首维不一致而报错；即使修正尺寸，也必须先将同一图窗口的N个节点序列梯度合并，再做一次整图裁剪。
- `stfednet_gated`除同一GRU问题外，GAT含有自定义参数和额外 `edge_index` 输入。Opacus默认functorch fallback未传入该图参数，在反向采样时以缺少 `edge_index` 报错，需要自定义GradSampler并明确每窗口跨所有节点的梯度汇总。
- `PrivacyEngine(secure_mode=True)`在当前Windows环境因没有 `torchcsprng` 而无法初始化。

所以Opacus不能直接接入现有模型；若正式G3采用它，需要实现DPGRU权重/状态适配、图窗口级GRU梯度归并及GAT GradSampler，并验证与原生模型/checkpoint的严格兼容性。当前没有实施这些改动。

### `torch.func`

`functional_call + grad`单样本在两种模型均可执行。当前PyTorch 2.10环境下，CPU `vmap(grad)`在GRU反向遇到batch维广播形状错误；CUDA在cuDNN启用时不能转换该GRU，关闭cuDNN后两种模型均可执行，并与循环剪裁梯度数值对齐。PyTorch提示 `aten::_thnn_fused_gru_cell` 使用慢速fallback batching rule，因此它是候选加速路线，仍需跨版本及正式batch配置验证。该方法需要保留 `[B, 参数量]`逐样本梯度堆栈，显存高于循环。

### 显式逐窗口循环

这是当前确认可正确处理两种正式模型的路线。测试验证：每窗口损失mean与`masked_huber`一致；所有参数的窗口梯度平均与确定性整batch梯度一致；先逐窗口裁剪再聚合的数值正确；裁剪后每记录范数不大于C；无噪声时聚合正确；带噪张量形状、总和级噪声方差和归一化尺度正确；不完整batch按实际条数处理，Poisson空batch能进行噪声更新。39、50、56、80、76节点图均可通过两种模型前后向计算。

## 会计器、抽样与预算示例

当前G2的 `DataLoader(shuffle=True)`每轮生成固定大小随机打乱batch，不是独立Poisson抽样。因此，**不能把以下Poisson-RDP数值套用到当前G2训练日志或现有checkpoint**。

如果正式G3明确改用与会计假设一致的Poisson采样，则可用成熟RDP Accountant按客户端累计：`n=6084`、期望batch `B=32`、采样率 `q=B/n=0.0052597`、每个名义本地epoch `ceil(n/B)=191`步、20轮且每轮1个本地epoch，共3820步/客户端；客户端参与率5/5；`delta=1e-5`。Poisson抽样是每步对每条窗口独立Bernoulli采样，实际batch可变，噪声按期望batch大小归一化。示例裁剪阈值取 `C=1`，但RDP表以noise multiplier `sigma`参数化，绝对噪声标准差是 `sigma*C`。

Opacus 1.6.0 RDP Accountant在这些**假设性采样/步数**下给出的示例（不是本地实际DP训练结果）：

| Noise multiplier σ | 每客户端示例 ε | δ |
|---:|---:|---:|
| 0.5 | 15.6347 | 1e-5 |
| 1.0 | 1.9793 | 1e-5 |
| 2.0 | 0.6936 | 1e-5 |
| 4.0 | 0.3078 | 1e-5 |

ε的差异来自噪声、采样率和总步数；这些数值不能作为实现可行性保证。正式协议需固定裁剪阈值、噪声乘数、采样策略、round/local epoch数、δ、客户端数据量和accountant版本，并为每个客户端单独记账。若同一底层曲线跨客户端重复出现，另做跨客户端组合分析。

可复现的普通Seed仅用于调试。正式DP噪声必须使用经审查的CSPRNG/secure RNG，并确保随机状态、种子和噪声样本不会发布给潜在对手。当前Opacus secure RNG缺少`torchcsprng`依赖，尚未在本机建立正式随机源；不能使用本项目固定Seed产生的普通`torch.randn`噪声宣称安全保护。

## CUDA短基准

基准使用真实客户端的训练切分，每方法取batch size 32、从同一个初始权重开始，普通路径采用AdamW+现有梯度范数裁剪，逐窗口和`torch.func`路径采用AdamW更新但不加噪声；每项预热后取3次中位数。结果含在 [`benchmark.json`](../results/g3_0/benchmark.json)；原始兼容性及预算记录含在 [`compatibility.json`](../results/g3_0/compatibility.json)，合并摘要在 [`feasibility.json`](../results/g3_0/feasibility.json)。测试切分设置了禁止访问对象，未用任何测试窗口。

| 模型 / 网 | 普通Batch32耗时 | 窗口循环Batch32耗时 | CUDA `torch.func` Batch32耗时 | 循环 / func峰值显存 | 普通epoch估计 | 循环 / func epoch估计 |
|---|---:|---:|---:|---:|---:|---:|
| GRU+Attention / 39 | 0.0036 s | 0.1029 s | 0.0848 s | 30.9 / 268.3 MB | 0.7 s | 19.6 / 16.2 s |
| ST-FedNet Gated / 39 | 0.0087 s | 0.1662 s | 0.0901 s | 32.9 / 430.3 MB | 1.7 s | 31.6 / 17.2 s |
| GRU+Attention / 80 | 0.0080 s | 0.0967 s | 0.0798 s | 37.7 / 535.1 MB | 1.5 s | 18.4 / 15.2 s |
| ST-FedNet Gated / 80 | 0.0266 s | 0.1595 s | 0.0911 s | 44.3 / 1064.4 MB | 5.1 s | 30.3 / 17.4 s |
| GRU+Attention / 76工业网 | 0.0162 s | 0.0979 s | 0.0836 s | 37.0 / 506.4 MB | 3.1 s | 18.6 / 16.0 s |
| ST-FedNet Gated / 76工业网 | 0.0218 s | 0.1607 s | 0.0912 s | 43.1 / 987.9 MB | 4.2 s | 30.6 / 17.4 s |

六种情况均完成，没有OOM。逐窗口循环批次约为普通训练的6至28倍耗时，估计完整本地epoch约18至32秒。CUDA `torch.func`约15至18秒/epoch，显存峰值约268至1064MB，显著高于逐窗口循环，但在当前Batch32条件下可用；它仍触发GRU慢速fallback提示。以上使用RTX 4060 Laptop独占的小型工程基准，不包含PyCharm其他GPU任务和噪声创建，正式部署前需在实际运行配置重测。

`grad_sample`缓存的理论张量规模（FP32、batch32）为：GRU+Attention单窗口17,924字节、批量堆栈573,568字节；ST-FedNet Gated单窗口18,580字节、批量堆栈594,560字节。逐窗口循环只累积一份全模型梯度；`torch.func`需要批量梯度堆栈。实际峰值还包括GAT/GRU中间激活、优化器状态及运行时缓存。所有峰值为PyTorch `max_memory_allocated`。epoch估计由一个限制batch线性外推到6084训练窗口，不含数据读取、验证、联邦通信、进程启动和高斯噪声创建；它是初步成本估计，不替代正式epoch计时。

## G3.0验收与G3实现建议

G3.0新增测试14项全部通过；全仓回归测试138项通过（包含既有G0至G2.2测试）。CUDA测试包含两个模型的 `torch.func` 与逐窗口循环梯度逐参数对照，以及训练Dropout可执行性检查。没有执行完整DP本地epoch、20轮联邦训练、正式测试评价或epsilon效用实验。

正式G3预期新增DP客户端更新模块与Poisson采样/会计配置，服务器继续FedAvg、但只接收客户端本地完成裁剪和加噪后的模型更新。建议先将循环作为严格数值oracle，以CUDA `torch.func`在CUDNN关闭条件下作为性能候选；Opacus需完成图窗口级自定义采样器后才可考虑。还需额外完成：

1. 明确并实现add/remove或replace-one邻接、C及noise multiplier定义；
2. 使用Poisson采样，并保证会计步数、空batch处理、期望batch归一化与训练实现一致；
3. 选择正式secure RNG，确保Seed/噪声不会暴露；
4. 固定或DP化Scaler、验证选择、公开指标以及拓扑/mask边界；
5. 针对两模型、五种节点数、CUDA/CPU、断点恢复和客户端组合进行回归；
6. 将每个客户端的累计ε、δ、q、步数、C、σ和会计器版本写入不可歧义的实验记录。

只有这些工作经独立验收后，才能在严格限定的窗口级邻接范围内讨论DP保证。
