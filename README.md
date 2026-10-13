# FedSTA-LoadForecasting

**基于时空注意力机制与联邦学习的配电网负荷预测**

G0 数据工程和 G1 预测基线、ST-FedNet 已实现。原始数据位于 `Datasets/`，五个客户端支持不同节点数。G1 使用 2021 年数据、48 小时输入、未来 1 小时预测；G0 的可配置 48→24 接口保留。

支持 Persistence、Seasonal Naive、GRU、GRU + Temporal Attention、GCN + GRU、GAT + GRU、ST-FedNet。ST-FedNet 使用双头 GAT → 单层 GRU → Additive Temporal Attention → Linear。

在项目根目录的 PyCharm PowerShell 终端执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --model stfednet --client lv_39 --epochs 30
.\.venv\Scripts\python.exe -m scripts.evaluate_model --checkpoint "<运行目录>/best.pt"
.\.venv\Scripts\python.exe -m scripts.train_model --resume "<运行目录>/latest.pt" --epochs 50
```

训练只读取训练/验证窗口，验证 WAPE 选择最佳 checkpoint；独立测试反归一化计算 MAE、RMSE、WAPE、MAPE。权重和预测数组保存在本地，日志及紧凑结果纳入版本管理。

- [G1 运行指南](docs/G1_RUN_GUIDE.md)：环境、七种方法、训练恢复、测试、结果汇总命令。
- [G1 验收报告](docs/G1_ACCEPTANCE_REPORT.md)：实际环境、自动化测试及小规模运行证据。
- [G1.1 技术说明](docs/G1_1_TECHNICAL_NOTE.md)：可学习空间残差、五客户端训练及验证集比较流程。
- [G0 验收报告](docs/G0_ACCEPTANCE_REPORT.md)与[数据说明](docs/data_description.md)：数据解析和质量处理。

后续 G2 开发 FedAvg/FedProx，G3 开发 DP-SGD。本阶段未执行正式长时间训练。

G1.1 新增模型可用以下命令在五客户端上依次训练，并与已有验证结果比较；批量训练只读取训练集和验证集，不执行测试集评估：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_clients --model stfednet_residual --output-root results/g1_1
.\.venv\Scripts\python.exe -m scripts.compare_validation --g1-root results/g1 --g11-root results/g1_1 --seed 42
```
