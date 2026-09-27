# v0.2.1 验证记录

日期：2026 年 9 月 27 日。

## 本次验证

- 在未安装 OpenAI SDK 的环境中安装主项目和 dev 依赖，运行 **25 项核心测试，全部通过**。
- 普通 CLI 帮助不显示 agent，主 Colab 没有外部模型 API 或密钥设置。
- 默认 pytest 只运行 tests；可选控制模块测试位于 optional_tests，不参与主安装。
- 教师学生训练、损失、数据处理与评价算法沿用 v0.2。
- 模拟 Colab 的文件上传和本地保存路径，实际执行新版 notebook 第 1、3、4、5、6 步的核心代码，
  完成原始合成 counts、预处理、四教师、四学生、五组实验、验证报告。第 2 步的安装和测试在此前独立执行。
- 合成 smoke 为 96 genes、20 targets、4 contexts，教师和学生各最多 5 epoch。
- 核心测试覆盖四学生损失梯度、外部教师缓存、续训、测试真值隔离、仅对照推理和 notebook 语法。
- 16 页中文 Word 指南经过渲染及逐页版式检查；同内容 Markdown 位于 docs/STRATEGY_AND_HOWTO_ZH.md。

## 实际环境

Python 3.12；PyTorch 2.14.0+cu130，测试设备为 CPU；NumPy 2.3.5；
pandas 2.2.3；SciPy 1.18.1；AnnData 0.12.19；pytest 9.1.1；nbformat 5.11.1。

## 尚未执行

真实 Colab 托管 GPU 环境、完整 Replogle/Nadig 训练、实际 GitHub 推送及在线模型 API。
本次没有重新运行可选控制模块测试，其说明单独保留在 docs/advanced/AGENT.md。

本次合成短训练没有显示对比损失收益，mean_transfer 仍更强。
docs/qa 的报告来自本次 notebook 核心流程验证，不构成真实生物数据上的性能证据。

