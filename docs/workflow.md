# 开发与实验约定

## 数据

- `data/raw/`：原始数据，只读使用，永不覆盖。
- `data/processed/`：预处理产物。
- `data/priors/`：GRN、转录因子等先验。
- 大型数据和产物保留在 AutoDL，不进入 Git。
- 数据集选定后记录来源、版本、预处理步骤和划分策略，防止数据泄漏。

## 执行顺序

LOCAL：实现代码 → 静态检查 / 小测试 → 检查 Git diff → commit → push。

AUTODL：git pull → 检查 GPU 环境 → 小规模 GPU smoke test → 检查形状、
损失、显存和产物 → 再开始完整训练。

`configs/smoke_test.yaml` 使用 1 epoch、batch size 8、最多 1000 cells、
2 workers 和 CUDA。`full_train.yaml` 使用完整数据；其 epoch、batch size、
worker 数量是初始建议值，需根据实际数据和显存调整。

## 后续训练实现要求

每次运行使用唯一 run ID，并分别保存：

- `checkpoints/<run_id>/`：模型、优化器状态、epoch、随机状态，支持恢复。
- `logs/<run_id>/`：完整实际配置、超参数、Git commit、环境版本、训练日志。
- `outputs/<run_id>/`：预测和评估指标。

统一由项目根目录解析配置中的相对路径，固定随机种子，训练与评估分开。
这些是实现约定，当前骨架尚未提供相应逻辑。

完整训练放在 tmux 会话中；按 Ctrl+B 后按 D 分离，
用 `tmux attach -t vcell` 恢复查看。
