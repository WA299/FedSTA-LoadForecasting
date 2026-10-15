# G3.2 DP 梯度信噪比诊断

## 目的与边界

G3.2 增加独立只读脚本 `scripts/diagnose_dp_gradient_snr.py`，检查 G3.1 三个 `stfednet_gated` 正式研究运行的 Best Checkpoint：σ=0、σ=0.5 和 σ=1.0。脚本不创建优化器、不执行模型参数更新、不调用正式DP训练循环，也不进行验证或测试集推理。预测网络、客户端训练及会计代码均未修改。

诊断以一个完整图时序窗口为记录，在每客户端TRAIN时间段选取三个固定、等距的32窗口批次；同一客户端的批次索引在三个Checkpoint间相同。为避免TEST接触，脚本直接拟合TRAIN时段Scaler，并且只变换TRAIN行、只建立TRAIN窗口视图；不创建验证或测试窗口Dataset。Checkpoint间只比较已保存的数据签名，不重新计算覆盖完整时序（包括测试时段）的数据哈希。

该脚本的数值、梯度统计、噪声扰动和Checkpoint验证WAPE均属于**内部限制访问的非DP产物**。详细机器结果写入 `results/g3_2/gradient_diagnostics.json`；`results/g3_2/` 已加入 `.gitignore`，结果文件不推送到GitHub。报告及代码不保存窗口索引、窗口内容、梯度/噪声张量或随机数生成器状态。随机源为普通PyTorch PRNG，仅用于研究诊断，不具备安全随机性或隐私声明效力。

## 计算口径

每批先调用 `src/privacy/per_sample.py` 中经G3.1验证的 `torch_func_clipped_gradients`（CUDA）；CPU使用显式逐窗口循环 `per_sample_clipped_gradients`。保持G3.1的 `C=1`、Huber目标、有效负荷节点掩码、全模型参数L2裁剪和期望Batch除数32。对第 *i* 条记录先裁剪 *g_i*，然后计算：

```text
g_bar = (1 / 32) * sum_i clip(g_i, C)
```

信号范数是这个**裁剪后Batch聚合平均梯度**的全参数L2范数。逐窗口裁剪前范数仅用于逐窗口平均范数、离散程度和裁剪比例；它不会被用作SNR分子。对每个噪声场景独立抽取高斯向量，均值梯度坐标的理论标准差为 `sigma*C/32`。报告实际采样噪声范数、实际坐标标准差、

```text
SNR = ||g_bar||_2 / ||noise||_2
cosine = cos(g_bar, g_bar + noise)
```

并对全模型及模块分组计算。模块包括GAT空间注意力、空间门控、空间输出投影、GRU、加性时间注意力和预测头。三批次的均值及样本标准差分别保存；每客户端保留其三个批次汇总。

诊断使用训练时模式，因此现有Dropout仍然生效；同一组固定窗口在不同测量中仍会经历随机Dropout掩码。重复批次离散程度因此同时包含有限窗口差异、Dropout差异和模拟噪声差异。噪声不会应用到模型或优化器，只参与临时向量度量。

## 运行与结果保护

在项目根目录使用既有 `.venv`：

```powershell
& 'E:\Dachuang_FedSTA\.venv\Scripts\python.exe' -m scripts.diagnose_dp_gradient_snr --device cuda --repeats 3
```

脚本默认读取三组正式Best Checkpoint并核对其训练噪声、20轮预算、Seed、裁剪阈值、Batch大小、客户端顺序、数据签名及 `smoke/test_evaluated/privacy_claim_allowed` 标记。它输出 `smoke=true`、`diagnostic_only=true`、`parameter_updates=false`、`test_evaluated=false`。本机RTX 4060 Laptop GPU采用CUDA `torch.func` 路径；PyTorch可能提示原生GRU batching-rule的性能回退，该警告不改变梯度数学口径。脚本不在不支持时自动切换后端。

将详细JSON限制在本地访问控制范围内，不要上传、公开或将其转成含指标的公共附件。代码和本技术说明只包含实现定义，不包含具体诊断标量。已有正式训练日志中的验证WAPE用于描述问题背景，但不是本诊断的测试指标，也不能视为DP保护的公开输出。

## 解释限制与建议

这些测量是在各运行的验证最佳权重上，用有限TRAIN窗口测量当前梯度，并对它们添加独立模拟噪声。它们能判断单步梯度方向与给定噪声量级的关系，不能证明20轮训练变差完全由噪声造成：训练过程中噪声累积、AdamW的自适应状态、联邦聚合、初始化差异、Dropout以及Checkpoint所处的局部区域都会影响最终WAPE。特别是一个停滞模型的当前梯度本来就可能较小，低SNR也可能是停滞的结果。

定量结果及根据预先声明条件生成的内部解释仅保留在上述限制访问JSON，不复制到GitHub文档。解读时优先看σ=0.5与σ=1.0正式权重上的全模型Batch均值SNR、方向余弦及其批次间变化，并确认各客户端和模块是否一致；不要仅凭逐窗口梯度范数、噪声标准差或一个Batch作因果判断。因低SNR也可能是训练停滞的结果，本文件不将有限Batch诊断解释为训练变差的因果证明。

下一项单变量实验可优先考虑把期望Batch大小提高至64，再考虑改变AdamW。该变化会使每步平均噪声标准差减半，但每轮名义更新步数约减半、抽样率和隐私会计也会变化，需记录并解释更新预算差异。详细结果及是否支持噪声主导假设留在本地限制访问JSON中。

本阶段没有访问TEST、没有执行长训练，也没有产生或声称任何差分隐私保证。
