> Historical v0.2.1 Colab walkthrough. For the current server workflow, see [SERVER.md](../SERVER.md).

# VCell 教师学生实验策略与部署指南

版本 v0.2.1　更新日期 2026 年 9 月 27 日

这份指南带你完成第一轮四教师、四学生实验：理解模型在预测什么，在 Colab 跑通代码，准备真实数据，读取对照结果，保存实验并推送源码。当前先专注教师学生训练，默认流程不需要外部模型 API。

**第一轮的目标是跑通实验并判断每一项训练约束是否有帮助。** 今天先完成 notebook 第 1 至 6 步。能看到完整验证报告后，再进入真实数据部分。

## 1 学习和执行顺序

| 阶段 | 你要做的事 | 完成时应得到什么 |
|---|---|---|
| 理解任务 | 读本指南第 2 至 5 节 | 能说清输入、输出和主要对照 |
| 软件测试 | notebook 第 1 至 6 步 | 合成数据报告和模型文件 |
| 真实试跑 | notebook 第 7 至 10 步 | 1 个 seed 的五组实验 |
| 确认结果 | 3 个 seed，固定方案再评分 | 验证及独立测试结果 |
| 保存迁移 | notebook 第 13 和 14 步 | Drive 结果及 GitHub 源码 |

## 先认识这些词

| 术语 | 在本项目中的具体含义 |
|---|---|
| 教师 teacher | 先训练、再提供预测目标的较宽网络 |
| 学生 student | 接受真实监督和可选额外约束的较窄网络 |
| 蒸馏 distillation | 让学生接近教师的预测；真实标签仍参与训练 |
| 互学 mutual learning | 学生的表达预测彼此提供额外训练目标 |
| 对比学习 contrastive learning | 拉近同一输入的跨学生表示，区分近似负对 |
| epoch | 遍历一遍训练组；不是运行一次代码格 |
| seed | 控制初始化和随机顺序的编号，用于检查波动 |
| checkpoint | 保存的模型状态；best.pt 用于推理，last.pt 用于续训 |

这里的教师与学生是训练角色，不是模型天然具有的身份。规模较大的教师也可能学得不好；学生是否受益要由实验决定。

<!-- PAGEBREAK -->

# 2 模型究竟在预测什么

给模型一个细胞背景的对照平均表达，以及一个已知扰动靶基因，模型预测该扰动在这个背景下引起的基因平均表达变化。例如，用某背景的非靶向对照细胞和靶基因 PERT001，预测干预后的平均表达相对对照会改变多少。

## 一条训练样本的含义

预处理后的每一行对应一个 dataset、一个 batch、一个 perturbation 的细胞组。原始 h5ad 中的一万多个细胞，聚合后可能只形成几百条训练样本。因此，原始细胞数量不能直接当成神经网络训练样本数量。

| 项目 | 假设有 2000 个输出基因 |
|---|---|
| baseline | 匹配对照的平均表达，长度 2000 |
| pert_idx | 扰动靶基因在训练词表中的编号 |
| delta 真值 | 扰动组均值减去匹配对照均值，长度 2000 |
| 模型预测 | 对 delta 的估计，长度 2000 |
| 重建的扰动均值 | baseline 加上预测 delta |

举一个示意例子：某基因的对照均值为 1.2，扰动均值为 1.5，则 delta 为 0.3。模型预测 0.2 时，对应扰动均值估计为 1.4。这里都是归一化后 log1p 表达的均值，不是原始 RNA 分子计数。

## 预处理的实际顺序

1. 对每个细胞，用全基因 counts 总量归一化到 target_sum，默认 10000。
2. 对归一化值做 log1p，再选择输出基因。
3. 按 dataset、batch、扰动聚合；对照也在同一 dataset 和 batch 内聚合。
4. 扰动组均值减去对应对照均值，得到训练目标。

基因方差排名来自训练细胞。所有分区只在基因注释层面确定共同基因集合。默认保留 2000 个高方差基因，并按训练细胞中的扰动频次最多选 200 个靶基因。

## 当前实验覆盖到哪里

这是“训练见过的扰动，在新细胞背景中的均值预测”。扰动通过可学习的编号嵌入表示，因此没有训练过的靶基因不能直接预测。当前不生成单细胞分布，也不输出正式比赛提交文件。

你原先考虑的 bulk 基线、TF 先验和调控网络还没有实现。此处 baseline 来自单细胞对照均值，mean_transfer 来自训练背景的真实扰动均值差。先把这一层评估清楚，再决定是否增加生物学模块。

<!-- PAGEBREAK -->

# 3 四个教师和四个学生怎样工作

教师和学生使用相同的四类结构，但隐层宽度不同。真实实验配置中，教师 hidden 为 256，学生为 64；合成 smoke 中分别为 64 和 32。每一个对应的学生版本都比同结构教师窄。

训练框架是 PyTorch，负责张量、网络、自动求导和 AdamW 参数更新。AnnData 读取 h5ad，NumPy 和 pandas 组织矩阵与注释，SciPy 拟合小型集成权重。Colab notebook 负责调用这些 Python 模块。

| 配置中的 architecture | 结构含义 | 想捕捉的关系 |
|---|---|---|
| residual | 背景和扰动交互后经过残差 MLP | 较复杂的非线性响应 |
| module | 8 个潜在模块 token 加扰动 token，经 Transformer | 不同潜在表达模块之间的交互 |
| mlp | 拼接背景表示和扰动表示，再经过 MLP | 较直接的条件映射 |
| bilinear | 背景表示与扰动表示做低秩乘性交互 | 相同扰动在不同背景中的变化 |

MLP 是多层感知机；Transformer 使用注意力混合 token 信息。这里的 8 个模块是网络学习的潜在表示，并不是事先定义的通路，也不是已验证的 TF 调控网络。

## 每个 seed 内的训练顺序

第一步，分别训练四个教师。参数更新只用训练分区的真实 delta，验证分区用于早停和 checkpoint 选择。

第二步，固定教师参数，生成预测。固定实验的蒸馏目标是四个教师预测的等权平均。报告中的 teachers/valmix 是另一项验证加权评估结果，**不会自动替换蒸馏所用的等权教师目标**。

第三步，依次运行五种学生实验。每一种实验都从相同的学生初始化开始，不接着前一种模式继续训练。四个学生在同一训练步骤中计算损失、一起反向传播。

第四步，输出各学生、等权集成和验证加权集成。每个 seed 独立执行上述流程；默认真实确认配置使用三个 seed。

同一模式的学生按等权集成验证 MSE 共同选择一个最佳 epoch，再拟合 valmix 权重；不是给四个学生分别挑四个不同 epoch。

## 多个模型是否同时占用多张 GPU

代码使用单设备训练。教师逐个训练；学生在一个训练循环内依次前向计算，并在同一步更新。它没有启动四个独立训练进程，也不需要四张 GPU。

“竞争”目前指验证排名及最终集成权重的选择。某学生可以得到零集成权重，但训练过程中不会被自动淘汰或复制。四个结构的差异只能提供互补的可能性，不能保证集成更准确。

<!-- PAGEBREAK -->

# 4 蒸馏 互学和对比学习分别做什么

先把真实监督当作所有实验的共同基础，再理解三项附加损失。它们约束的位置不同，不能仅因为总 loss 下降，就认为扰动预测更准确。

| 损失 | 比较什么 | 作用位置 |
|---|---|---|
| supervised | 学生预测与真实 delta | 输出表达变化 |
| kd | 学生预测与四教师等权预测 | 输出表达变化 |
| peer | 两个学生的预测 | 输出表达变化 |
| contrastive | 两个学生的投影向量 | 潜在表示 |

总损失可以读成：

真实监督 + kd_weight × 蒸馏损失 + ramp × peer_weight × 互学损失 + ramp × contrast_weight × 对比损失。

## 教师蒸馏

假设真实 delta 是 0.3，教师平均预测是 0.25。学生同时向真实结果和教师结果靠近。教师较平滑的预测可能帮助学生，也可能把错误传给学生，所以必须与只用真实监督的学生比较。

## 学生互学

每个学生把另一个学生当前的预测当成一个暂时固定的目标。stop-gradient 表示计算这一方向的损失时，不沿目标一侧继续传播梯度；另一方向有对应的损失。

四个学生一共有六对。代码对六对损失取平均，不会因为学生数量增加就把同一权重机械放大六倍。

## 跨学生对比学习

同一条输入在两个学生中的表示构成正对。不同扰动的其他输入作为近似负对；同一扰动的其他行不作为负对。投影头只是训练表示的辅助分支，最终评价仍看表达预测。

不同扰动可能影响同一条通路，因此“不同扰动”并不等于“生物学不相似”。对比损失是否适合这个数据，需要由 contrastive 与 kd 的比较回答。

## 训练日志中的数值

delta 先除以训练集计算出的一个 RMS 尺度，再计算训练损失。history.csv 中的 loss 和 val_normalized_mse 使用这个标准化尺度；summary.csv 中的 mse_delta_mean 使用原始 delta 尺度。不要把两份文件里的数值直接相减。

warmup_epochs 期间不启用互学和对比项；之后用五个 epoch 逐步把 ramp 增加到 1。真实默认 warmup 为 5；smoke 为 1。

<!-- PAGEBREAK -->

# 5 第一轮实验策略

第一轮固定模型数量、结构、数据划分和训练预算，只比较训练约束。这样你才能回答“是蒸馏有用，还是对比有用”，而不是把所有改动混在一起。

| 模式 | 真实监督 | 蒸馏 | 互学 | 对比 |
|---|---|---|---|---|
| supervised | 有 | 无 | 无 | 无 |
| kd | 有 | 有 | 无 | 无 |
| mutual | 有 | 有 | 有 | 无 |
| contrastive | 有 | 有 | 无 | 有 |
| mutual_contrastive | 有 | 有 | 有 | 有 |

## 预先确定主要比较

主要比较设为 **contrastive/mean 对 kd/mean**，检验在蒸馏基础上增加对比学习是否有帮助。两者都使用四学生等权集成，因此不会额外混入混合权重拟合的影响。

辅助比较包括 kd/mean 对 supervised/mean、mutual/mean 对 kd/mean，以及 mutual_contrastive/mean 对 mutual/mean。每个学生的个体分数也要看；不能用一个强学生掩盖其他学生全部退化。

## 三个阶段的预算

| 阶段 | seeds | 教师上限 | 学生上限 | patience | 用途 |
|---|---|---|---|---|---|
| 合成 smoke | [0] | 5 epoch | 5 epoch | 5 | 检查软件流程 |
| 真实 pilot | [0] | 20 epoch | 20 epoch | 6 | 检查数据和初步趋势 |
| 真实确认 | [0, 1, 2] | 100 epoch | 80 epoch | 15 | 检查结果的重复性 |

pilot 使用 real.yaml 的结构和损失参数，只缩短 epoch 上限并减少 seed。短试跑可能尚未收敛，不能因为一轮 pilot 没提升就宣布方法无效。

每个 seed 有 4 次教师训练和 5 组四学生训练，共 24 个独立模型训练实例。三个 seed 共 72 个实例；这是模型数，不是 72 个进程。patience 触发时会提前结束。

本轮不同时搜索模型数量。四模型是否优于两模型是另一个对照问题，不能从四模型结果本身推出。先保持四教师四学生，获得可解释的第一轮结果。

<!-- PAGEBREAK -->

# 6 在 Colab 完成安装和保存设置

你需要配套的 VCell_TeacherStudent_Colab.ipynb 和 VCell_TeacherStudent_v0.2.1.zip。notebook 是操作界面，ZIP 是真正的代码、配置和测试。

## 对应 notebook 第 1 步

打开 Google Colab，在文件菜单选择上传 notebook，上传 ipynb。然后运行第 1 个代码格，在出现的文件选择框中上传 ZIP。这里上传的是 ZIP，不要再上传 notebook，也不用提前解压。

该格会建立 PROJECT 变量并切换工作目录。看到一个包含 vcell_dual 的路径，说明代码解压完成。

## 对应 notebook 第 2 步

可以在运行时菜单更改运行时类型并选择 GPU。CPU 也能运行合成测试。安装格实际上执行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

-e 表示从当前源码目录安装项目；[dev] 增加测试和 notebook 所需依赖。成功后应看到 25 passed。CUDA: True 表示 PyTorch 当前能看到 GPU；选了 GPU 运行时本身并不等于训练已经在使用它。[1]

## 对应 notebook 第 3 步

保留 USE_DRIVE=True，按提示挂载 Drive。WORK 默认设置为：

```text
/content/drive/MyDrive/VCell_TeacherStudent_v021
```

| 变量或位置 | 存放什么 | 是否需要长期保存 |
|---|---|---|
| PROJECT | 解压的源码 | 可从 ZIP 或 GitHub 重建 |
| WORK/configs | 你的实际实验配置 | 需要 |
| WORK/demo 或 prepared_real | 准备好的数据 | 需要 |
| WORK/smoke_01 或 real_pilot_01 | 训练权重和报告 | 需要 |

Colab 虚拟机可能因空闲或生命周期限制被删除；保存 notebook 并不保存虚拟机里的全部文件。[1] 此处将结果直接写入 Drive，优先保证首轮实验容易恢复。

<!-- PAGEBREAK -->

# 7 跑通合成实验并找到输出

## 对应 notebook 第 4 步

运行合成数据格。它先生成四个背景的人工 counts h5ad，再调用正式的数据预处理函数。默认得到 96 个基因、20 个扰动，每个背景两个 batch 的组级数据。

先查看 metadata.csv 的头几行。确认其中有 context、batch、perturbation、split、n_cells 和 n_controls。再看按 split/context 统计的组数，理解模型究竟读了多少条样本。

## 对应 notebook 第 5 步

直接运行训练格。等价命令是：

```bash
python -m vcell run --config configs/smoke.yaml
```

notebook 额外传入了 Drive 下的数据和输出路径。日志先出现 t_residual、t_module、t_mlp、t_bilinear，再出现五个学生模式。每行的 epoch、loss、val 分别表示训练轮次、总训练损失和等权集成的标准化验证误差。

已完成后再次运行同一格，会使用 --resume 读取已完成的 checkpoint，而不是把它当作新的独立重复。

## 对应 notebook 第 6 步

| 文件 | 第一次查看时关注什么 |
|---|---|
| evaluation_validation/report.html | 实验是否全部出现，指标是否完整 |
| summary.csv | 个体、mean、valmix 的验证 MSE |
| paired_comparisons.csv | 候选与参照的 MSE 差值 |
| diversity.csv | 学生对的误差相关性和预测分歧 |
| seed_0/contrastive/history.csv | 四项损失、ramp 和验证轨迹 |
| seed_0/contrastive/best.pt | 可用于推理的已训练模型 |

这时先停止，不必下载真实数据。你已经完成了环境安装、预处理、四教师四学生训练、早停、集成和报告生成。

随包合成 QA 的短训练中，mean_transfer 比神经网络更强，对比学习也没有显示收益。这个结果保留在报告中。合成实验的通过标准是流程正确、输出可解释，不是某一种新损失必须胜出。

<!-- PAGEBREAK -->

# 8 真实数据和数据划分

真实数据使用带有扰动标签、对照标签、batch 和基因注释的 counts h5ad。当前代码没有内置通用的 GEO MTX 转换器；只有 MTX 时，需要先按原始 barcode 和 guide 注释整理成正确的 AnnData。

| 背景 | 默认分区 | 数据入口 |
|---|---|---|
| K562 | train | Replogle 作者发布数据 [3] |
| RPE1 | train | Replogle 作者发布数据 [3] |
| HepG2 | val | GSE264667 [4] |
| Jurkat | test | GSE264667 [4] |

这是一套开发用划分，不是对官方比赛分区的复现。不同背景的数据处理、实验技术和批次也可能不同，跨背景表现会同时受到这些因素影响。

## 对应 notebook 第 7 步

已有 h5ad 时，填 H5AD_TO_INSPECT 后运行。没有文件时可先开启 LIST_FILES=True，列出 Figshare 文件目录，再选择具体文件 ID。DOWNLOAD_FILE_ID 只填你确认的文件，不要下载全部 article。

```bash
python -m vcell catalogue
python -m vcell inspect /实际路径/k562.h5ad
```

inspect 会显示矩阵维度、obs 列名、layers、部分标签值和 gene ID 示例。它不证明某层一定是 counts，也只展示部分标签；遇到 control 未出现在示例中，应进一步查看该列取值。

## 需要核对的内容

- 表达矩阵：.X 或某一 layers 键中是否保留原始、非负整数 counts。
- 扰动列：是靶基因、guide ID，还是组合扰动；若是 guide，先正确映射到靶基因。
- 对照标签：实际字符串是什么，不能直接照抄 non-targeting。
- batch 列：能否在同一批次内找到匹配对照。
- 基因标识：所有文件使用可对齐的 gene ID，且选定标识没有重复。

Arc 的合并数据入口也可用于寻找文件，但当前页面缺少 dataset card，不能据此认定其矩阵已经满足本包的 counts 输入要求。[5]

先处理这些注释，再训练模型。GPU 无法修正错误的对照标签、重复 gene ID 或错误的 guide 映射。

<!-- PAGEBREAK -->

# 9 填写数据配置

notebook 第 8 步会把模板写入 WORK/configs/data_real.yaml，并设置 prepared_real 输出目录。你在 Colab 左侧文件浏览器打开这个 YAML 编辑，然后保存。

下面是结构示例；路径和字段值都要按自己的文件修改。使用绝对路径最容易避免“相对于哪个目录”的混淆。

```yaml
output_dir: /content/drive/MyDrive/VCell_TeacherStudent_v021/prepared_real
train_contexts: [K562, RPE1]
val_contexts: [HepG2]
test_contexts: [Jurkat]
target_sum: 10000
max_genes: 2000
max_perturbations: 200
min_cells: 20
min_control_cells: 30
chunk_size: 2048
seed: 17
datasets:
  - id: k562
    path: /你的目录/k562.h5ad
    context: K562
    gene_key: null
    perturbation_key: gene
    batch_key: gem_group
    count_layer: X
    control_values: [non-targeting]
  - id: rpe1
    path: /你的目录/rpe1.h5ad
    context: RPE1
    gene_key: null
    perturbation_key: gene
    batch_key: gem_group
    count_layer: X
    control_values: [non-targeting]
  - id: hepg2
    path: /你的目录/hepg2.h5ad
    context: HepG2
    gene_key: null
    perturbation_key: gene
    batch_key: gem_group
    count_layer: X
    control_values: [non-targeting]
  - id: jurkat
    path: /你的目录/jurkat.h5ad
    context: Jurkat
    gene_key: null
    perturbation_key: gene
    batch_key: gem_group
    count_layer: X
    control_values: [non-targeting]
```

gene_key: null 表示使用 var_names。count_layer: X 表示使用 .X；如实际 counts 在 layers["counts"]，填 counts。min_cells 和 min_control_cells 是每个组的门槛，不是整个数据集的总数。

<!-- PAGEBREAK -->

# 10 从预处理进入真实试跑

## 对应 notebook 第 9 步

核对配置后，将 PREPARE_REAL 改为 True。prepare 会生成 dataset.npz、metadata.csv 和 data_audit.json。它会排除细胞数不足、缺少匹配对照，以及训练词表中没有出现的扰动组。

看 data_audit.json 中的 split_counts、n_rows、n_genes、n_perturbations 和 skipped_groups。train、val、test 都必须有有效组；不能只看到程序生成文件就直接往下训练。

若多个背景在同一个 h5ad，可给每条 datasets 配置相同路径但不同的 row_filter，例如：

```yaml
row_filter:
  cell_line: k562
```

其中 cell_line 和 k562 必须是实际 obs 列及取值；不同 entry 的行不能重叠。数据配置中的相对路径以 YAML 所在目录为起点。

## 对应 notebook 第 10 步

首轮保持这些值，再将 RUN_REAL 改为 True：

```python
RUN_NAME = 'real_pilot_01'
SEEDS = [0]
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 20
PATIENCE = 6
```

该格从 configs/real.yaml 读取网络和损失设置，写出你的完整训练配置，再执行训练。准备好的数据目录应与 REAL_DATA 一致。运行完后 ACTIVE_RUN 会切换为 REAL_RUN，你可以回到第 6 步看完整报告。

## 什么时候进入三个 seed

确认分区、对照、基因顺序和各模式输出都正确后，用新 RUN_NAME，例如 real_confirm_01，设置 SEEDS=[0,1,2]、教师 100 epoch、学生 80 epoch、patience=15。

同一套五组实验都使用这些设置。实际 epoch 数会因验证早停而不同，但上限和选择规则相同。不要只给表现好的模式更多训练时间，然后把改善归因于它的损失函数。

原始 h5ad 预处理主要涉及 CPU、内存和磁盘。输出基因降到 2000 并不意味着原始文件读取量也缩小到相同比例；程序仍需计算全基因细胞总量。

<!-- PAGEBREAK -->

# 11 怎样改参数而不失去可比性

第一轮先使用默认损失权重。看到实际问题后，再提出一个明确的改动，例如“教师太弱，降低蒸馏强度”或“训练和验证都未收敛，延长所有模式的训练上限”。

| 参数 | real 默认值 | 你需要理解的作用 |
|---|---|---|
| learning_rate | 0.0005 | 每次优化更新的尺度 |
| weight_decay | 0.0001 | 对参数大小施加正则约束 |
| batch_size | 64 | 每批组级样本数，也影响可用负对 |
| kd_weight | 0.3 | 学生接近教师平均预测的强度 |
| peer_weight | 0.1 | 学生预测互相靠近的强度 |
| contrast_weight | 0.02 | 跨学生表示对比的强度 |
| temperature | 0.2 | 对比相似度分布的尖锐程度 |
| warmup_epochs | 5 | 前期只做监督和可选蒸馏 |
| patience | 15 | 验证无改善多少 epoch 后停止 |

## 一个可解释的第二轮实验

如果 kd 比 supervised 更差，先看 teachers/mean 的分数和四个教师的个体分数。若教师平均预测不够好，可以只把 kd_weight 从 0.3 调成 0.1。保留数据划分、seed、结构和其他参数，新建输出目录重新跑五组对照。

如果只想测试对比强度，将 contrast_weight 从 0.02 调成 0.01，并记录这一个变化。它只在包含对比损失的模式中生效。暂时不要同时改 hidden、batch_size 和多个权重，否则难以解释结果来源。

## 修改文件的方法

复制 configs/real.yaml 或 WORK 中已有训练 YAML，修改参数并指定新的 output_dir。然后在 PROJECT 目录运行：

```bash
python -m vcell run --config /你的目录/train_round02.yaml
```

训练 YAML 里的相对 data_dir、output_dir 和 cache 路径，以执行命令时的工作目录为起点；这和数据 manifest 的相对路径规则不同。教程中实际数据和结果都使用绝对路径。

同一次 run 内教师会被五组学生实验复用；换成新的 output_dir 会重新训练教师。这版没有跨实验目录的自动教师缓存管理。先以清楚、可复现的比较为目标，不为尚未发生的开销增加调度层。

<!-- PAGEBREAK -->

# 12 如何判断对比学习有没有效果

主要看真实数据上的 delta 预测误差，再看重复性与模型间关系。表中的 model 名称例如 contrastive/s_mlp 是个体学生，contrastive/mean 是等权集成，contrastive/valmix 是验证集拟合的非负加权集成。

两个简单基线也要看：no_change 把所有 delta 预测为零；mean_transfer 对每个扰动先平均训练背景内的 batch，再平均背景，直接将该效应迁移到新背景。神经网络应和这些基线比较，不能只在五种神经网络模式中选冠军。

## 先看差值方向

paired_comparisons.csv 中的 delta_mse 是 candidate 减 reference。假设 kd/mean 的 MSE 是 0.12，contrastive/mean 是 0.11，则差值是 -0.01，候选更好。这个数字例子只用于理解方向。

1. 查看 contrastive/mean 对 kd/mean 是否降低 MSE。
2. 查看各 seed 是否方向一致，而不是只看平均数。
3. 查看不同学生是否多数受益，还是只靠一个模型。
4. 查看 mutual_contrastive/mean 对 mutual/mean 是否也出现类似方向。
5. 查看误差相关性时同时看 MSE；更不一致的模型也可能只是更差。

valmix 的权重在验证集上拟合；即使验证集表现更好，也需要独立测试。等权集成用于主要方法比较，valmix 用于辅助观察是否有可利用的互补性。

Pearson 反映基因变化模式的相关程度，不能替代误差大小。top-k overlap 和 direction 是诊断指标，不是正式差异表达显著性检验。

## 误差和区间怎样汇总

报告先平均同一扰动的 batch 组，再平均扰动和背景，最后汇总 seed。mse_delta_std 是 seed 间标准差；只有一个 seed 时为空，不是零不确定性。

bootstrap 按扰动重采样，先平均 seed 和 batch。区间只针对当前评估背景的扰动集合，不能当成所有未知细胞类型的总体区间。验证集还参与过早停和权重选择，因此验证结果存在选择偏差。

## 最后才评价测试集

方案固定后，在 notebook 第 11 步设 REVEAL_TEST=True。程序会报告所有模式，但主要比较仍按事先确定的 contrastive/mean 对 kd/mean 解读，不根据测试排行榜重新选方案。

如果根据 Jurkat 测试结果继续调参，Jurkat 就已参与开发。后续确认需要独立背景或新的预先固定评估设计。现阶段可以报告“在当前留出背景上的表现”，不要直接扩大成跨所有细胞类型有效。

<!-- PAGEBREAK -->

# 13 新背景推理和断点恢复

## 仅用对照预测新背景

notebook 第 12 步使用已训练的 best.pt。先按验证结果确定 SELECTED_MODE，再准备 query YAML 和 targets.csv。这个步骤不要求新背景的扰动后表达。

targets.csv 有一列 perturbation，列中写训练词表已经出现的靶基因。query 配置参考 configs/query.example.yaml，实际输入只需包含对照细胞，例如：

```yaml
min_control_cells: 30
chunk_size: 2048
datasets:
  - id: new_controls
    path: /你的目录/controls.h5ad
    context: new_context
    gene_key: null
    perturbation_key: gene
    batch_key: batch
    count_layer: X
    control_values: [control]
```

只含对照细胞的文件也需要正确的扰动标签列，例如每行 gene 都标为 control。模型会按 batch 对照均值和每个待预测靶基因生成输出。

mean_predictions.npz 包含 delta、baseline、predicted_mean 和各学生预测。predicted_mean 是 baseline 加预测 delta。checkpoint 中已固定学生混合权重，推理阶段不再用新背景真值选权重。

## 训练中断后怎么继续

1. 用同一新版 ZIP 重新打开 notebook，完成上传和安装。
2. 重新挂载同一个 Drive，设置相同 WORK。
3. 保持同一份准备数据、源码和训练配置。
4. 运行原训练格；已有 run_manifest 时，notebook 会加 --resume。

last.pt 包含优化器和训练状态。best.pt 保存已经选定的最佳 epoch，用于预测。一个 epoch 完成后会更新续训文件；若在 epoch 中间中断，该 epoch 的未保存部分会重做。

修改源码、配置、数据或教师缓存后，用新的输出目录。这次 v0.2.1 的入口和源码有更新，请给新试验使用新目录。不同硬件或依赖版本不保证逐位一致，不要把换卡续训当作新的独立 seed。

<!-- PAGEBREAK -->

# 14 保存工作和推送 GitHub

## 三类东西分别保存

| 内容 | 保存位置 | 作用 |
|---|---|---|
| 源码和默认配置 | ZIP 及 GitHub | 能重新安装、检查和修改实现 |
| 实际数据与实验 YAML | Drive 的 WORK | 能重建这一次实验 |
| 权重、运行清单与报告 | WORK 下的运行目录 | 能续训、推理、解释结果 |

notebook 第 13 步可把验证报告打包下载。完整训练成果仍以 WORK 中的数据、配置和运行目录为准。当前源码推送脚本不会把 WORK 中的实际训练 YAML 自动放进 Git；可以额外保存一份去掉私人路径的配置，但原件仍留在 Drive。

## 对应 notebook 第 14 步

1. 在 GitHub 创建自己的空仓库，首次不要预建 README，避免远端已经有另一条提交历史。
2. 填 GITHUB_REPO、GIT_NAME 和 GIT_EMAIL。可使用 GitHub 提供的 noreply 邮箱。
3. 创建仅针对该仓库的 fine-grained personal access token。
4. Contents 选择 Read and write。本包包含 .github/workflows，推送该文件还需 Workflows 写权限。[2]
5. 将 DO_PUSH 改为 True，运行代码格，在隐藏输入框输入 token。

脚本提交源码白名单，随后推送到 main。数据、权重和运行结果不在提交中。这个脚本适合首次推入空仓库；之后如果 GitHub 已有新的提交，应先 clone 当前仓库再更新代码，不要每次解压 ZIP 后重新 init 推送。

## 以后移到本地或 AutoDL

从你自己的仓库 clone 后，在仓库根目录安装同一项目：

```bash
git clone https://github.com/你的用户名/你的仓库.git
cd 你的仓库
python -m pip install -e '.[dev]'
python -m pytest -q
python -m vcell run --config /你的目录/train.yaml
```

数据和结果需要另外复制；Git clone 不会带上 Drive 中的 checkpoint。复制准备数据时一起带上 dataset.npz、metadata.csv 和 data_audit.json。先确认新环境能读配置和完成 smoke，再继续长训练。

普通 Python/终端运行上面的命令即可；notebook 中的 google.colab 上传和挂载单元只用于 Colab。

<!-- PAGEBREAK -->

# 15 代码模块和你现在要做的事

| 文件或目录 | 负责什么 | 当前是否需要修改 |
|---|---|---|
| configs/smoke.yaml | 合成软件测试预算 | 第一遍不改 |
| configs/data.example.yaml | 真实 counts 的来源与字段 | 复制后按文件填写 |
| configs/real.yaml | 网络列表、损失和训练参数 | 首轮沿用 |
| src/vcell/data.py | 归一化、组均值、分区和准备数据 | 先读逻辑 |
| src/vcell/models.py | 四种网络结构及输出头 | 暂时不改结构 |
| src/vcell/losses.py | 两两互学与对比损失 | 对照结果出来后再研究 |
| src/vcell/train.py | 训练、早停、恢复与模式顺序 | 先沿用 |
| src/vcell/selection.py | 宏平均误差与集成权重 | 先沿用 |
| src/vcell/evaluate.py | 指标、成对比较和 HTML 报告 | 先学会读取 |
| src/vcell/inference.py | 从对照细胞预测新背景 | 训练完成后使用 |
| tests/test_core.py | 教师学生流程的 25 项检查 | 安装后执行 |


今天先完成 notebook 第 1 至 6 步，并确认你能找到 report.html、history.csv、best.pt 和 last.pt。下一步整理四个真实文件的路径、counts 层、扰动列、batch 列和 control 标签，再进入第 7 至 10 步。

<!-- PAGEBREAK -->

# 16 常见疑问和参考资料



**为什么 GPU 利用率不高？** 原始细胞已聚合成少量组，网络和 batch 可能很小；预处理、读取文件和保存 checkpoint 也主要依赖 CPU 或磁盘。先查看组数、CUDA 状态和每个 epoch 耗时，不能仅靠 GPU 利用率判断代码是否错误。Drive 的频繁读写也可能增加耗时。[1]

**看到路径错误或没有对照怎么办？** 先核对 PROJECT、WORK、manifest 文件位置和实际 obs 标签。inspect 只显示部分示例；需要时读取对应列的 value_counts。按事实修正配置，再创建新的准备目录。

**某个学生特别差是否立即删除？** 第一轮保留它，并检查个体误差和集成权重。删除学生会改变实验条件，应作为下一次明确对照，不要在同一次运行中随意改变成员。

## 官方资料

[1] Google Colab FAQ：https://research.google.com/colaboratory/faq.html

[2] GitHub token 与仓库文件权限：
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
https://docs.github.com/en/rest/repos/contents

[3] Replogle 数据：https://doi.org/10.25452/figshare.plus.20029387

[4] HepG2 与 Jurkat 数据入口：https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667

[5] Arc 合并数据入口：https://huggingface.co/datasets/arcinstitute/State-Replogle-Filtered

平台操作与权限说明核查于 2026 年 9 月 27 日。模型行为以本包 v0.2.1 源码为准。真实生物数据、Colab GPU 实机和 GitHub 推送仍需在你的环境执行。
