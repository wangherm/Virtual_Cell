# VCell 教师学生实验
v0.2.1 · 四个参考教师、四个学生、五组固定对照。

先读 [实验策略与部署指南](docs/STRATEGY_AND_HOWTO_ZH.md)，再打开
`notebooks/VCell_TeacherStudent_Colab.ipynb`。第一次只运行 notebook 第 1 至 6 步。

## 这一轮做什么

使用对照平均表达和扰动 ID，预测跨背景的平均表达变化。教师从头训练；
目前不加载现成基础模型权重，也不生成单细胞分布或正式比赛提交文件。

| 实验 | 真实监督 | 蒸馏 | 互学 | 对比 |
|---|---|---|---|---|
| supervised | 有 | 无 | 无 | 无 |
| kd | 有 | 有 | 无 | 无 |
| mutual | 有 | 有 | 有 | 无 |
| contrastive | 有 | 有 | 无 | 有 |
| mutual_contrastive | 有 | 有 | 有 | 有 |

主比较：contrastive/mean 对 kd/mean。先固定 4 教师和 4 学生，不同时搜索数量。
数据、结构和损失实现沿用 v0.2；本次整理部署入口和教学文档。

## Colab

1. 上传 `VCell_TeacherStudent_Colab.ipynb`。
2. 第 1 格上传 `VCell_TeacherStudent_v0.2.1.zip`。
3. 安装后默认运行 25 项核心测试。
4. 挂载 Drive，生成合成数据，运行五组实验，读取验证报告。
5. 按指南检查真实 h5ad 并编辑数据 YAML，再手动开启真实训练。

## 本地或服务器

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m vcell demo --output data/demo
python -m vcell run --config configs/smoke.yaml
```

CPU 环境可先从 PyTorch 官方 CPU 索引安装 torch，再安装本项目。
Colab 安装使用其已有的兼容 PyTorch 环境。

默认安装和核心测试不依赖外部模型 API。可选扩展独立保留，不参与以上流程。

## 数据与结果

真实数据模板：`configs/data.example.yaml`。先运行 inspect 检查文件，不照抄字段。
模型参数：`configs/real.yaml`，默认三个 seed。
所有真实标签仅在所属分区使用：train 更新参数，val 早停和选权重，test 最后评价。

结果包含 `evaluation_validation/report.html`、`summary.csv`、`paired_comparisons.csv`，
以及 `seed_*/MODE/best.pt` 和 `last.pt`。
同一版本、配置和准备数据中断续训时加 `--resume`。
新版本实验使用新输出目录，不覆盖旧运行。

[外部教师缓存](docs/EXTERNAL_TEACHERS.md)　[验证记录](docs/VALIDATION.md)　[方法与来源](docs/REFERENCES.md)

代码 MIT；外部数据和权重遵循各自许可。

