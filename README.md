# Virtual Cell

单细胞扰动响应预测研究项目。Windows / VS Code 本地开发，通过私有 GitHub
仓库同步代码，AutoDL 负责 GPU 执行。

当前是项目骨架：尚未实现数据加载、模型、训练或评估。实验配置是待接入的模板。

```text
Virtual_Cell/
├── src/virtual_cell/
│   ├── data/           # 数据读取、预处理、划分
│   ├── models/         # 基线、编码器、扰动响应模型
│   ├── priors/         # GRN、转录因子等先验
│   ├── training/       # 训练循环、损失、checkpoint 恢复
│   ├── evaluation/     # 预测、指标、分布评估
│   └── utils/          # 配置、路径、随机种子等公共功能
├── configs/
│   ├── smoke_test.yaml
│   └── full_train.yaml
├── scripts/            # 工作目录初始化与 GPU 环境检查
├── tests/              # 后续添加不依赖真实数据的小测试
├── notebooks/          # 探索分析；提交前清除大型输出
├── docs/               # 工作流与数据约定
├── data/               # Git 忽略；raw / processed / priors
├── checkpoints/        # Git 忽略
├── outputs/            # Git 忽略
├── logs/               # Git 忽略
├── AGENTS.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## LOCAL：Windows PowerShell

```powershell
cd "$HOME\Desktop\Virtual_Cell"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/init_workspace.py
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

本地基础环境不安装 PyTorch，也不需要 GPU 或大数据。
依赖目前是基础版本范围；模型选定、AutoDL 环境验证后再记录确切版本。

首次推送：先在 GitHub 创建 **Private** 空仓库，不勾选自动生成 README、
License 或 .gitignore，然后执行（替换仓库地址）：

```powershell
git init -b main
git status
git add .
git diff --cached --stat
git diff --cached
git commit -m "Initialize Virtual Cell project structure"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

如果 Git 要求身份信息，先设置自己的 `git config user.name` 和
`git config user.email`。认证使用 GitHub 支持的 SSH 或令牌方式，勿提交凭据。

## AUTODL：SSH 登录后

```bash
cd /root/autodl-tmp
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git virtual_cell
cd virtual_cell
python -m pip install -r requirements.txt
python scripts/init_workspace.py
nvidia-smi
python scripts/check_gpu.py
```

使用已配置 CUDA 版 PyTorch 的环境，或按服务器 CUDA 环境单独安装 PyTorch。
`check_gpu.py` 成功时打印 PyTorch 版本、`CUDA available: True` 和 GPU 名称。
缺少 PyTorch 或 CUDA 不可用时返回非零退出码。
**这只是环境检查，不是模型训练 smoke test；当前没有训练命令。**

后续同步代码用 `git pull --ff-only`。正式训练前先实现并通过
`configs/smoke_test.yaml` 对应的小规模实验；长任务使用 `tmux new -s vcell`。

Git 不保存空目录或被忽略的运行目录，因此每次 clone 后运行
`python scripts/init_workspace.py`。详细约定见 [工作流](docs/workflow.md)。

下一步：确定首个数据集、输入表达矩阵格式、扰动标签及训练/验证/测试划分，
再实现一个简单基线和真实的 GPU smoke test。
