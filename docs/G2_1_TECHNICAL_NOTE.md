# G2.1 FedProx 训练诊断与近端系数敏感性支持

项目：**基于时空注意力机制与联邦学习的配电网负荷预测**。基于 G2 提交 `3e64a6f22735808b87fc584a7cce86b3d52ebf85` 开发，仅增加观测性诊断和验证结果分析。预测架构、FedAvg 加权、FedProx 数学目标、每轮客户端 AdamW 重新初始化、随机种子、Batch 顺序、梯度裁剪及五网等权验证 WAPE 选优规则不变。Centralized Round-Robin、Local-only、G0/G1/G1.1/G1.2 保留。

## 诊断计算与平均口径

客户端基于所有实际执行的训练 Batch 统计，不读取验证/测试数据来计算训练诊断。每个 Batch 的 Huber 仍仅包含有效负荷节点，其内部归约保持原实现；诊断值是在标准化空间下的训练目标，不能当作 kW 预测误差。

| 字段 | 含义 | 客户端轮内平均 | 服务器轮内平均 |
|---|---|---|---|
| `huber_loss` / `mean_huber_loss` | 原 Huber 数据拟合项 | 按 Batch 实际窗口数加权 | 按客户端实际处理窗口数加权 |
| `prox_loss` / `mean_prox_loss` | `mu/2 * sum((w_local-w_round_start)^2)` 的实际贡献 | 按窗口数加权 | 按窗口数加权 |
| `total_loss` / `mean_total_loss` | Huber + Prox | 按窗口数加权 | 按窗口数加权 |
| `grad_norm_before_clip` / `mean_grad_norm_before_clip` | 总目标反传后、裁剪前全体可训练参数梯度的 L2 范数 | 实际 Batch 算术平均 | 按实际 Batch 数加权 |
| `clip_fraction` | 既有裁剪系数小于 1 的 Batch 占比 | 裁剪 Batch 数 / 全部 Batch 数 | 合计裁剪 Batch 数 / 合计 Batch 数 |

`local_epochs` 的重复访问会计入实际窗口数/Batch 数；最后不足一个 Batch 的窗口按实际数量参与损失平均。梯度范数和裁剪比例不使用节点数、窗口数或客户端等权平均。FedAvg 与 FedProx `mu=0` 的 Prox 为 0；两者诊断和更新完全相同。

近端项与总目标在 `fedprox_objective(..., return_prox=True)` 内一次计算；默认调用仍返回原来的总目标，`mu=0` 默认仍直接返回原 Huber Tensor。客户端复用原来唯一一次 `clip_grad_norm_` 返回的裁剪前范数，按其既有 `max_norm/(norm+1e-6)` 系数记录裁剪。没有增加梯度缩放、反传、预测或随机采样。

原 `TrainLoss` / `train_loss` / `mean_loss` 的含义完全保留：按实际处理窗口加权的总本地训练目标，FedProx 包含近端项。新 `mean_total_loss` 与它一致。近端目标是参数平方距离的总和，不按参数数目或节点数平均。

## 日志、持久化与恢复

FedAvg/FedProx `rounds.csv` 追加 `mean_huber_loss`、`mean_prox_loss`、`mean_total_loss`、`mean_grad_norm_before_clip`、`clip_fraction`、`client_diagnostics_json`。每轮 JSON 汇总为 `round_diagnostics.json`，包含五客户端统计、处理窗口数、Batch 数、裁剪 Batch 数及平均口径。终端和 `train.log` 增加 Huber、Prox、Total、GradNorm、ClipFraction；Prox 使用六位有效数字以保留较小数值。

Latest 额外保存 `current_diagnostics`、`best_diagnostics`；Best 对应最佳验证轮次的诊断。原完整轮次 checkpoint 字段和 RNG 恢复流程保留。恢复时重新构建完整轮次 CSV/诊断 JSON，修复未提交输出；混合旧/新列使用字段并集，不能丢弃新字段。旧权重缺少诊断时，旧轮次 JSON 为 `null`、CSV 空白，比较器标记 `not recorded`（未记录），不得用误差反推历史训练诊断。若最佳轮次仍是旧轮次，其最佳诊断也保持未记录。

## 多 mu 比较

`scripts/compare_fedprox_mu.py` 是独立入口，不改变 `compare_g2.py`。它读取 Best/Latest checkpoint 内已有的验证摘要，不构建数据集或执行模型前向。默认保留完整的 20 轮基准，按算法/模型/Seed/mu 分组：

1. 排除 smoke、Batch 受限运行、缺失 Latest 或未完成配置轮数的运行，报告跳过原因。
2. 重复运行按 Latest 文件的 `mtime_ns` 最大值选择，再以绝对路径字典序确定并列；不根据 WAPE 选择重复实验。所有符合条件的重复运行都保留在 `all_runs` 清单中，原文件不修改。
3. 校验同模型和 Seed、五客户端列表/数据签名、48→1、70/15/15，以及 hidden/dropout/heads、Batch/LR/weight decay、Huber/MAPE/gradient clip、参与率、轮次/local_epochs/worker 数与 Batch 限制。FedProx 只允许 mu 不同，配置与 checkpoint 的 mu 必须一致。
4. 校验对应五客户端等权 WAPE 与 Best 记录、最佳轮次及训练完成预算。每组与匹配 FedAvg 比较；没有对应 FedAvg 或协议不一致时拒绝比较。
5. 输出每客户端 WAPE、等权与 pooled WAPE、最佳轮次、相对 FedAvg 的百分点变化、配置、签名、训练预算、诊断是否记录，保存 JSON/CSV。默认扫描所有 Seed，`--seed 42` 限定当前公平比较组。

目前只读识别的正式结果：

| 模型 | FedAvg 最佳 WAPE | 最佳轮次 | FedProx mu=0.01 最佳 WAPE | 最佳轮次 |
|---|---:|---:|---:|---:|
| `stfednet_gated` | 24.2258% | 18 | 29.0460% | 19 |
| `gru_attention` | 23.5756% | 20 | 28.9917% | 19 |

mu=0.001/0.0001 尚未正式执行，比较 JSON 明确列出四项缺失的计划实验。历史 mu=0.01 的诊断为未记录，不能据此断言其近端项过大或算法有误。

## FedAvg 收敛诊断

`scripts/analyze_fedavg_convergence.py` 仅读取被选中正式运行的验证 `rounds.csv` 和 Best/Latest 元数据。校验 CSV 轮次连续、无遗漏/重复、未包含 smoke，五网 WAPE 与等权 WAPE 相符、CSV 最佳轮次与 Best 相符。输出全轮曲线、第 15～20 轮、最后五轮各客户端变化、最后一轮是否最佳，以及线性斜率/极差。

描述性平台判据：最后五轮斜率绝对值 ≤0.05 个百分点/轮且极差 ≤0.5 个百分点；同时展示最后三轮，避免较早的改善掩盖近期趋平。该启发式不是收敛证明，不凭“最后一轮最佳”判断未收敛。

| 模型 | 第15轮→第20轮 WAPE | 15→20 净变化 | 最后五轮净变化 | 最后一轮最佳 |
|---|---|---:|---:|---|
| `gru_attention` | 23.9370% → 23.5756% | -0.3613 个百分点 | -0.0952 个百分点 | 是 |
| `stfednet_gated` | 26.2233% → 24.2634% | -1.9599 个百分点 | -1.3877 个百分点 | 否，最佳第18轮 |

GRU+Attention 最近五轮斜率为 -0.0160 个百分点/轮、极差 0.1782，近平台；尽管第20轮最佳，不能据此认定未收敛。Gated 最近五轮仍有下降趋势，但第18～20轮已趋平；可考虑独立的40轮延续以澄清，不保证额外轮数能改善。各客户端最后五轮详细数值见 `results/g2_1/fedavg_convergence.json`，CSV 保存全部逐轮曲线。

本次没有启动延长训练。未来若决定延续至40轮，先把完整20轮目录复制到新的目录，再在副本恢复；原 Best/Latest、配置、日志和20轮基准不得被覆盖。分析默认 `--rounds 20`，避免混入延长预算。

## PyCharm 正式实验命令

在 `E:\Dachuang_FedSTA` 终端执行。使用原 `.venv` 和 `configs/g2.yaml`，明确固定 Seed、轮数、local_epochs、Batch 和 LR，其他配置保持原值；每次自动建立唯一的新运行目录，不覆盖 mu=0.01。

```powershell
.\.venv\Scripts\python.exe -m scripts.train_federated --config configs/g2.yaml --algorithm fedprox --model stfednet_gated --mu 0.001 --seed 42 --rounds 20 --local-epochs 1 --batch-size 32 --learning-rate 0.001 --device cuda
.\.venv\Scripts\python.exe -m scripts.train_federated --config configs/g2.yaml --algorithm fedprox --model stfednet_gated --mu 0.0001 --seed 42 --rounds 20 --local-epochs 1 --batch-size 32 --learning-rate 0.001 --device cuda
.\.venv\Scripts\python.exe -m scripts.train_federated --config configs/g2.yaml --algorithm fedprox --model gru_attention --mu 0.001 --seed 42 --rounds 20 --local-epochs 1 --batch-size 32 --learning-rate 0.001 --device cuda
.\.venv\Scripts\python.exe -m scripts.train_federated --config configs/g2.yaml --algorithm fedprox --model gru_attention --mu 0.0001 --seed 42 --rounds 20 --local-epochs 1 --batch-size 32 --learning-rate 0.001 --device cuda
```

汇总及只读收敛分析：

```powershell
.\.venv\Scripts\python.exe -m scripts.compare_fedprox_mu --root results/g2 --seed 42 --rounds 20 --output results/g2_1/fedprox_mu_comparison.json
.\.venv\Scripts\python.exe -m scripts.analyze_fedavg_convergence --root results/g2 --seed 42 --rounds 20 --output results/g2_1/fedavg_convergence.json
```

## 验收证据

- 完整回归 `116 passed`，原101项及新增15项全部通过。新增测试覆盖预 G2.1 冻结客户端循环的两模型×两算法参数/损失逐位相等、真实 mu 缩放项、末 Batch 的窗口加权、Batch 加权梯度统计与裁剪比例、旧 schema 恢复、缺失诊断、重复运行选择、mu 分组/协议拒绝与最终轮最佳但近平台的反例。FedAvg 与 mu=0 诊断同样逐项相等。
- 同环境 CUDA 短冒烟：RTX 4060 Laptop GPU、PyTorch 2.10.0+cu126，五网节点39/50/56/80/76，Batch2、每客户端两个训练 Batch/一个验证 Batch、两轮，两个模型×FedAvg/FedProx和两种Centralized对照全部完成。参数更新、严格重载、连续与恢复的权重/验证一致，FedAvg/FedProx诊断恢复一致。
- 六个CUDA组合与本地保存的 G2.1 前短冒烟权重逐项比较，参数、TrainLoss、验证结果均完全相同。8份旧正式 FedAvg/FedProx Best/Latest 严格加载，SHA256保持不变；实际旧权重没有恢复/延长写入，旧schema恢复在小规模合成训练中验证。
- 第二轮 FedProx 冒烟中，Gated Huber=0.324533692、Prox≈1.142949e-5、Total=0.324545123、GradNorm=0.510750183；GRU+Attention Huber=0.351724170、Prox≈1.112568e-5、Total=0.351735298、GradNorm=0.615649828；均无裁剪。这只是两个本地Batch的工程验证，不能解释整轮191个Batch的正式 mu=0.01 性能下降，也不是性能结论。
- 本地权重与完整调试输出在 `.pytest_tmp/g2_1_smoke/`；可版本管理证据在 `results/g2_1/cuda_smoke_summary.json`、`cuda_smoke.log`、`checkpoint_compatibility.json`、两份分析JSON/CSV。smoke与正式结果明确区分，测试 split 由守卫对象禁止访问。

本阶段没有执行正式20轮训练或测试集评估；未修改网络，未增加FedPer、FedAtt或差分隐私模块。
