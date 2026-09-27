# 逐项替换外部教师

默认四个教师都是自训参考模型。此接口接收你在外部模型官方环境生成的均值差预测，不负责安装或冒充 State、GEARS 等模型。

## 导出输入索引

```bash
python -m vcell queries --data data/prepared --output data/teacher_queries
```

输出对照 baseline、扰动 ID、gene IDs 和 row IDs，不含真实扰动表达。需要对照单细胞集合的模型应读取对应原始对照细胞，不能把平均值复制成假细胞输入。

## 生成和导入预测

`teacher_delta.npy` 的每行是：预测的平均 log1p(normalized-expression) 减对应对照的平均 log1p(normalized-expression)。目标尺度必须与本包相同。latent embedding、raw counts、logFC 不能直接用作 delta。

配套 CSV 分别有 `row_id` 和 `gene` 列，导入时按 ID 重排。每个请求行和目标基因需完整覆盖。

复制 `configs/teacher_provenance.example.json`，记录实际来源、revision、许可、预训练/微调接触过的扰动背景及明确排除的背景。声明会被检查，但程序无法证明预训练数据没有污染。

```bash
python -m vcell import-teacher \
  --data data/prepared --matrix data/external/teacher_delta.npy \
  --rows data/external/row_ids.csv --genes data/external/genes.csv \
  --provenance data/external/provenance.json --output data/teacher_external.npz
```

NPZ 和同名 JSON 一起保存。把配置列表中的任意项换成：

```yaml
teachers:
  - {name: t_external, cache: data/teacher_external.npz}
  - {name: t_module, architecture: module, hidden: 256, dropout: 0.1}
  - {name: t_mlp, architecture: mlp, hidden: 256, dropout: 0.1}
  - {name: t_bilinear, architecture: bilinear, hidden: 256, dropout: 0.1}
```

仅 cache 项跳过训练，其余照常；缓存也可全部替换。运行配置里的路径相对于当前工作目录。固定外部缓存跨 seed 共用，不把重复教师分数解释成独立重复。

来源未知或见过留出扰动标签的教师，不适用于此严格跨背景评估。查清来源后再导入；不要用验证/测试真值填补缺失输出。

