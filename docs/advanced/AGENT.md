# 可选训练控制模块

该模块暂不进入教师学生教学和部署流程。主安装 `.[dev]`、主 CLI 和默认测试均不依赖它。
实现保留在 src/vcell/agent.py，相关测试在 optional_tests/test_agent.py。
不提供旧的 `python -m vcell agent` 命令别名。

需要重新启用时，先完成固定教师学生实验，再执行：

```bash
python -m pip install -e '.[agent]'
python -m vcell.agent --run runs/real --model YOUR_API_MODEL --max-trials 3
```

模型需支持 Responses API Structured Outputs；密钥通过 OPENAI_API_KEY 环境变量提供。
中断后同一命令加 --resume。模块只读取专用验证摘要，调节学习率、weight decay、
蒸馏/互学/对比权重及教师混合权重；模型结构、seed、数据分区、epoch 上限固定。
教师预测复用，学生每轮从固定初始化重训。程序按验证分数选优；LLM 文字不能覆盖指标。

```bash
python -m pip install -e '.[dev,agent]'
python -m pytest -q optional_tests
```

v0.2 曾通过模拟决策和真实 SDK 的离线测试；本次只验证当前默认教师学生路径。
恢复研究该模块时再运行上述可选测试。在线 API 调用仍需单独验证。

