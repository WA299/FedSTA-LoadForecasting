"""Render the closure report from committed metric aggregates; never run models."""

import json
from pathlib import Path

import numpy as np

from src.fl.common import CLIENT_IDS
from src.training.final_evaluation import (PROJECT_NAME, atomic_json, canonical,
                                           save_results, sha256, summarize)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_results(result, manifest):
    import hashlib
    content = {k: v for k, v in manifest.items() if k != "freeze_sha256"}
    if hashlib.sha256(canonical(content)).hexdigest() != result["manifest_sha256"]:
        raise ValueError("Result/manifest binding mismatch")
    entries = {e["entry_id"]: e for e in manifest["entries"]}
    if (len(result["records"]) != len(entries) or
            {r["entry_id"] for r in result["records"]} != set(entries)):
        raise ValueError("Incomplete formal suite")
    for record in result["records"]:
        entry = entries[record["entry_id"]]
        for key in ("checkpoint", "config", "validation_metrics", "privacy", "data_signature"):
            if record[key] != entry[key]:
                raise ValueError("Frozen result metadata mismatch: " + key)
        if record["split"] != "test" or record["smoke"] or not record["model_state_unchanged"]:
            raise ValueError("Not a formal inference-only result")
        if record["test_windows"] != 1314:
            raise ValueError("Not full 2021 TEST coverage")
    if summarize(result["records"]) != result["methods"]:
        raise ValueError("Summary inconsistent with per-client sufficient statistics")


def validation_context():
    context = {"split": "validation", "test_evaluated": False, "sources": {}}
    paths = {"architecture": "results/g1_2/validation_comparison.json",
             "fedprox": "results/g2_1/fedprox_mu_comparison.json",
             "fedper": "results/g2_2/fedper_comparison.json",
             "batch64": "results/g3_3_batch64/G3_3_SUMMARY.json"}
    for key, name in paths.items():
        if not Path(name).exists():
            continue
        d = read_json(name)
        if d.get("test_evaluated"):
            raise ValueError("Historical context must be validation-only")
        context["sources"][key] = {"path": name, "sha256": sha256(name)}
        if key == "architecture":
            context[key] = d["client_equal_validation_wape_percent"]
        elif key in {"fedprox", "fedper"}:
            context[key] = [{k: r.get(k) for k in ("algorithm", "model_name", "mu", "best_round",
                              "client_equal_wape_percent")} for r in d["runs"]]
        else:
            context[key] = {k: d[k] for k in ("expected_batch_size", "noise_multiplier", "max_grad_norm",
                            "completed_rounds", "client_equal_validation_wape_percent")}
    return context


def main():
    root = Path("results/final_test")
    result = read_json(root / "final_test_metrics.json")
    manifest = read_json("configs/final_test_manifest.json")
    validate_results(result, manifest)
    context = validation_context()
    atomic_json(root / "validation_context.json", context)
    # Re-render summaries from stored results only; this does not repeat inference.
    save_results(root, manifest, result["records"])
    methods = result["methods"]
    records = result["records"]
    equal = lambda key: methods[key]["client_equal_wape_percent"]
    local, gru = "local_stfednet_gated", "local_gru_attention"
    full = {k: v for k, v in methods.items() if v["complete_five_clients"]}
    winner = min(full, key=lambda k: equal(k))
    delta = equal(local) - equal(gru)
    gated_wins = sum(methods[local]["clients"][c]["wape_percent"] < methods[gru]["clients"][c]["wape_percent"] for c in CLIENT_IDS)
    lines = ["# 算法探索与项目结项报告", "", PROJECT_NAME, "",
             "## 结项范围与结果来源", "",
             f"正式清单含{len({e['checkpoint']['sha256'] for e in manifest['entries']})}份Checkpoint、{len(records)}个模型—客户端组合；冻结哈希 `{manifest['freeze_sha256']}`。",
             "模型、Seed=42、数据签名、48→1任务、70/15/15切分及训练配置均在首次读取TEST前冻结。重复运行按Latest完成时间与路径选择；FedProx固定沿用G2.1验证敏感性选择μ=0.001。",
             "本次没有新训练、超参数搜索或测试后换模型。旧权重未修改。G3.3 Batch64、G3.4 10轮学习率及C=0.5探索不纳入最终主测试；FedPer仅引用已有验证结果。",
             "每组完整测试五张图的全部1314个目标窗口（2021年，精确起止时间见JSON），仅有效负荷母线参与指标；单位为kW。MAPE仅统计|真实值|≥0.1 kW，并报告有效比例。",
             "验证与测试分属不同季节的时间区间，因此测试WAPE低于验证WAPE本身不能解释为模型进一步进步；两类指标不得混作同一评估集比较。",
             "全局模型没有保存Scaler数组，评估恢复数据签名与预处理配置一致的本地Checkpoint中已有Scaler；未调用fit。源码数据全数组只用于已有签名校验，数据视图仅创建TEST及其必要历史上下文，不创建TRAIN/VAL窗口。",
             "Persistence只有lv_39正式权重，其余四网记为缺失，未补训；其单网指标不参与五网排行。所有完整预测数组、权重和随机状态继续留在本地，不提交。", "",
             "## 最终正式测试结果", "",
             "等权WAPE为五个客户端WAPE的算术平均；Overall WAPE由全部有效观测合并后的绝对误差之和除以真实负荷绝对值之和计算。工业网负荷较大，Overall与等权排行可能不同。",
             "", "| 方法 | 五网等权WAPE (%) | Overall WAPE (%) | 相对本地Gated变化 (百分点) |", "|---|---:|---:|---:|"]
    for key, group in full.items():
        lines.append(f"| {key} | {equal(key):.6f} | {group['overall']['wape_percent']:.6f} | {equal(key)-equal(local):+.6f} |")
    lines += ["", f"以预先采用的五网等权WAPE衡量，完整五网方法中最低者为 **{winner}：{equal(winner):.6f}%**。这只是当前冻结清单与单Seed数据上的描述，不是重新选择训练Checkpoint或声称统计显著。",
              "完整逐客户端MAE、RMSE、WAPE、MAPE与MAPE有效比例见 `results/final_test/FINAL_TEST_SUMMARY.md` 和同目录CSV/JSON。", "",
              "## 预测结构收益", "", "| 客户端 | GRU-Attention测试WAPE (%) | Gated测试WAPE (%) | Gated−GRU (百分点) |", "|---|---:|---:|---:|"]
    for cid in CLIENT_IDS:
        g, s = methods[gru]["clients"][cid]["wape_percent"], methods[local]["clients"][cid]["wape_percent"]
        lines.append(f"| {cid} | {g:.6f} | {s:.6f} | {s-g:+.6f} |")
    lines += ["", f"Gated相较GRU-Attention的等权测试WAPE变化为{delta:+.6f}个百分点（相对变化{100*delta/equal(gru):+.3f}%；负值表示误差降低），在{gated_wins}/5个客户端更低。",
              "结构仍为双头GAT→节点自适应小初值空间门控残差→单层GRU→单头加性时间注意力→Linear；节点数不影响参数形状。比较反映整套结构差异，不能仅凭门控大小证明真实物理拓扑贡献。",
              "已有验证对照（以下不是测试结果）：", "", "| 本地结构 | 五网等权验证WAPE (%) |", "|---|---:|"]
    for model, value in context.get("architecture", {}).items():
        lines.append(f"| {model} | {value:.6f} |")
    lines += ["", "空间残差和门控在部分客户端出现改善，但现有验证结果没有显示对纯时间注意力的稳定五网优势。空间结构复杂度增加不等于预测收益；最终只测试了预先冻结的GRU-Attention与Gated，不给原始/标量残差变体补造测试结果。", "",
              "## Centralized、Local-only与联邦方法", "",
              "Local-only是五套独立最佳模型；Centralized是单一模型与单一AdamW、跨客户端Round-Robin Batch交错；FedAvg/FedProx是同一轮聚合后的一套全局模型。不同节点数图分别推理，不拼接节点轴。",
              "", "| 方法 | lv_80测试WAPE (%) | 工业网测试WAPE (%) | 相对Local等权变化 (百分点) |", "|---|---:|---:|---:|"]
    for key in (local, "centralized", "fedavg", "fedprox"):
        c = methods[key]["clients"]
        lines.append(f"| {key} | {c['lv_80']['wape_percent']:.6f} | {c['industrial_mvlv']['wape_percent']:.6f} | {equal(key)-equal(local):+.6f} |")
    lines += ["", "联邦共享参数在异构负荷分布上可能出现负迁移，尤其需看80节点网和工业网，不能用工业网占比较大的Overall替代等权指标。独立本地模型最多训练30 Epoch并早停；联邦为20轮×本地1 Epoch；Centralized为20轮各客户端训练集均参与，优化器连续保留。这些不是严格匹配计算预算的因果对照。",
              "FedProx敏感性与FedPer历史验证结果（未追加测试）：", "", "| 方法 | 模型 | μ | 最佳轮次 | 五网等权验证WAPE (%) |", "|---|---|---:|---:|---:|"]
    rows = context.get("fedprox", []) + [r for r in context.get("fedper", []) if r["algorithm"] == "fedper"]
    for row in rows:
        lines.append(f"| {row['algorithm']} | {row['model_name']} | {row['mu']} | {row['best_round']} | {row['client_equal_wape_percent']:.6f} |")
    lines += ["", "μ=0.01的验证误差数值明显高于较小μ；GRU-Attention的μ=0.0001验证结果优于其FedAvg，而Gated所试μ仍未超过FedAvg。FedPer仅个性化Linear头，验证增益有限；五个head与共享骨干的组合不能当作FedAvg单全局模型。这里引用验证观察，不写成正式测试结论。", "",
              "## 隐私—效用", "",
              "三组均完成20轮、C=1、期望Batch=32、q=32/6084、每轮191步、Seed=42；同为Poisson抽样、逐窗口裁剪与AdamW。σ=0是相同机制的非DP控制，与原随机洗牌FedAvg协议不同。",
              "", "| σ | Best轮次 | 每客户端Best累计步数 | Best理论ε | δ | 验证等权WAPE (%) | 测试等权WAPE (%) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ("dp_sigma0", "dp_sigma0.5", "dp_sigma1"):
        row = next(r for r in records if r["method"] == key)
        account = row["privacy"]["best_checkpoint_accounting"]["lv_39"]
        eps = account["theoretical_epsilon"]
        eps_text = f"{eps:.6f}" if eps is not None else "不适用（非DP）"
        val = np.mean([r["validation_metrics"]["wape_percent"] for r in records if r["method"] == key])
        lines.append(f"| {account['noise_multiplier']:g} | {row['best_epoch_or_round']} | {account['steps']} | {eps_text} | {account['delta']:g} | {val:.6f} | {equal(key):.6f} |")
    noise_delta = equal("dp_sigma0.5") - equal("dp_sigma0")
    lines += ["", f"σ=0.5相对无噪声控制的等权测试误差变化为{noise_delta:+.6f}个百分点；更强噪声降低理论窗口级隐私损失，但在当前配置下未获得良好的误差—预算折中。",
              "已有G3.2有限只读诊断支持噪声主导的单步现象，但并非停滞的因果证明；限制访问的梯度数值、张量与随机状态不在本报告公开。"]
    if "batch64" in context:
        lines += [f"G3.3 Batch64探索的已有等权验证WAPE为{context['batch64']['client_equal_validation_wape_percent']:.6f}%，未显示优于Batch32 σ=1的明显改善。它未进入最终测试，也不用于重新选择主实验。"]
    lines += ["", "所有DP组均为research普通PRNG模式，Opacus RDP给出的ε仅在固定Scaler、拓扑和mask、add/remove完整窗口邻接、Poisson抽样等假设下解释。这里展示Best本身的账户，不用Latest20轮账户替代；Best ε也不能代表发布完整20轮轨迹与基于私有验证选优的端到端预算。",
              "滑动窗口重叠、跨客户端重复曲线使窗口级结论不能直接扩展到原始时间点、家庭、用户或客户端级隐私；五客户端各自会计不等于对跨客户端重复个体完成组合分析。训练集Scaler拟合、非DP验证选优、公开测试统计均未获得DP保护。本次按结项要求公开的是非DP汇总指标，不公开权重或随机状态。DP-SGD也不自动提供安全聚合、通信加密或身份隐藏。", "",
              "## 有效探索、限制与最终结论", "",
              "1. 统一数据工程、可变节点接口、严格时间划分、训练期Scaler恢复、验证选优和Checkpoint恢复使各阶段结果可核查；本次冻结与一次性测试补齐了最终评价闭环。",
              "2. GRU时间编码与加性时间注意力是稳健的轻量基线；GAT、空间残差及节点门控的收益具有客户端差异，不能声称空间模块稳定优越。",
              "3. FedAvg可共享跨图参数，但异构数据下存在负迁移；FedProx对μ敏感，Linear头FedPer未解决所有异构性问题。Centralized与Local表现必须按实际测试数据和预算解释，不预设集中式必胜。",
              "4. 窗口级逐样本裁剪、Poisson抽样、加噪与逐客户端RDP会计已实现工程可行性；当前研究噪声设置出现明显效用代价，不能据此声称端到端严格隐私系统已完成。",
              "5. 限制包括单一2021年数据、五张挪威参考图、单Seed、重叠窗口、部分重复曲线、LV单位为来源未标注情况下的MW→kW显式假设、不同训练预算及未经隐私化的预处理/模型选择。没有多Seed置信区间，不作统计显著性或泛化到所有配电网的断言。",
              "项目以当前冻结结果结项，不再启动训练、调参、FedAtt或新模型；测试结果仅用于最终报告。", "",
              "## 文件与复现核对", "",
              "冻结清单：`configs/final_test_manifest.json`；正式汇总：`results/final_test/final_test_metrics.json`、CSV与`FINAL_TEST_SUMMARY.md`；历史验证摘要与来源哈希：同目录`validation_context.json`。每条测试记录含Checkpoint路径/哈希、配置、Best轮次/epoch、Scaler来源、数据签名与MAPE有效比例。",
              "`scripts.evaluate_final_suite --plan-only`不加载源数据；`--execute --device cuda`只推理。已存在清单不允许静默覆盖。执行账本只跳过已完成组合；若异常发生在前向与指标原子提交之间，会拒绝自动重做该组合，避免冒充精确一次。",
              "`python -m scripts.report_final_suite`只读取汇总JSON重建文档与交叉核对，不加载模型或测试数据，不重复测试。单组合账本及临时锁保留本地，不提交。"]
    acceptance_path = root / "acceptance.json"
    if acceptance_path.exists():
        acceptance = read_json(acceptance_path)
        if acceptance["manifest_sha256"] != result["manifest_sha256"]:
            raise ValueError("Acceptance manifest mismatch")
        lines += ["", "## 最终工程验收", "",
                  f"全仓自动化测试{acceptance['automated_tests']['passed']}项通过，其中本次新增{acceptance['automated_tests']['new_tests']}项；10条既有torch.func GRU批处理性能警告不影响正确性。",
                  "RTX 4060 Laptop GPU完成41/41个正式测试组合，共53874次窗口推理；34个Best/Latest来源文件的冻结哈希全部保持一致。CSV的41行与JSON逐项一致，汇总由有效观测充分统计量交叉核对。详细验收元数据见 `results/final_test/acceptance.json`。"]
    path = Path("docs/ALGORITHM_EXPLORATION_REPORT.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Report] {path.resolve()} | checked {len(records)} formal TEST records")


if __name__ == "__main__":
    main()
