"""Generate the teacher-student teaching notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# VCell 教师学生实验与部署
v0.2.1 · 4 个参考教师、4 个学生、5 组固定实验。
先阅读 docs/STRATEGY_AND_HOWTO_ZH.md。第一次只运行第 1 至 6 步，即可完成合成数据测试。
默认流程不需要外部模型 API。真实数据和 GitHub 推送在后面的单元中手动开启。
模型预测平均表达变化；教师从头训练，当前不加载基础模型预训练权重。""")
md("## 1 上传代码包\n在 Colab 文件菜单上传本 notebook，然后在这一格选择配套 ZIP。ZIP 不用手动解压。")
code("""from pathlib import Path
import os, sys, subprocess, tempfile, zipfile, io, json
from google.colab import files
uploaded = files.upload()  # VCell_TeacherStudent_v0.2.1.zip
zip_names = [k for k in uploaded if k.lower().endswith('.zip')]
assert len(zip_names) == 1, '请只选择一个代码 ZIP'
staging = Path(tempfile.mkdtemp(prefix='vcell_'))
with zipfile.ZipFile(io.BytesIO(uploaded[zip_names[0]])) as archive:
    for member in archive.infolist():
        target = (staging / member.filename).resolve()
        if not target.is_relative_to(staging.resolve()):
            raise ValueError('Unsafe archive path')
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Symlink in archive')
    archive.extractall(staging)
PROJECT = staging / 'vcell_dual'
os.chdir(PROJECT)
del uploaded
print('代码目录:', PROJECT)""")
md("## 2 安装并检查\n运行时菜单可选择 GPU；合成测试也支持 CPU。测试成功应显示 25 passed。")
code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[dev]'], check=True)
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)""")
md("## 3 设置保存位置\n建议挂载 Drive。PROJECT 放代码，WORK 放数据、配置和结果。只保存 notebook 不能保存训练权重。")
code("""USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = Path('/content/drive/MyDrive/VCell_TeacherStudent_v021')
else:
    WORK = Path('/content/VCell_TeacherStudent_v021')
WORK.mkdir(parents=True, exist_ok=True)
(WORK / 'configs').mkdir(exist_ok=True)
def vcell(*args):
    subprocess.run([sys.executable, '-m', 'vcell', *map(str, args)], check=True)
print('数据和结果:', WORK)""")
md("## 4 生成合成数据并看训练样本\n先生成 counts，再走与真实数据相同的预处理。metadata 的一行是一组细胞，不是单个细胞。")
code("""import pandas as pd
from IPython.display import display, HTML
DEMO = WORK / 'demo'
if not (DEMO / 'prepared/dataset.npz').exists():
    vcell('demo', '--output', DEMO)
meta = pd.read_csv(DEMO / 'prepared/metadata.csv')
display(meta.head())
display(meta.groupby(['split', 'context']).size().rename('groups'))
display(json.loads((DEMO / 'prepared/data_audit.json').read_text()))""")
md("## 5 运行四教师四学生\nsmoke 每个模型最多 5 epoch、1 个 seed。先训练四个教师，再依次训练五组学生实验。")
code("""DEMO_RUN = WORK / 'smoke_01'
args = ['run', '--config', PROJECT / 'configs/smoke.yaml',
        '--data', DEMO / 'prepared', '--output', DEMO_RUN]
if (DEMO_RUN / 'run_manifest.json').exists():
    args.append('--resume')
vcell(*args)
ACTIVE_RUN = DEMO_RUN""")
md("## 6 读取验证报告\nMSE 越低越好。paired_comparisons 中 delta_mse 为候选减参照，负数有利于候选。合成结果只检验程序。")
code("""summary = pd.read_csv(ACTIVE_RUN / 'evaluation_validation/summary.csv')
display(summary[['split', 'model', 'mse_delta_mean', 'mse_delta_std']])
display(pd.read_csv(ACTIVE_RUN / 'evaluation_validation/paired_comparisons.csv'))
display(pd.read_csv(ACTIVE_RUN / 'evaluation_validation/diversity.csv').head(12))
display(pd.read_csv(ACTIVE_RUN / 'seed_0/contrastive/history.csv'))
display(HTML((ACTIVE_RUN / 'evaluation_validation/report.html').read_text()))""")
md("""## 7 选择真实数据文件
Replogle K562/RPE1：https://doi.org/10.25452/figshare.plus.20029387
HepG2/Jurkat：https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667
先列目录，再决定下载哪一个原始 counts 文件；不是下载全部。完整文件可能很大。
已有 h5ad 则跳过下载，直接填 H5AD_TO_INSPECT。""")
code("""LIST_FILES = False
if LIST_FILES:
    vcell('catalogue')
DOWNLOAD_FILE_ID = None
if DOWNLOAD_FILE_ID is not None:
    vcell('download', '--file-id', DOWNLOAD_FILE_ID, '--output', WORK / 'raw')
H5AD_TO_INSPECT = ''
if H5AD_TO_INSPECT:
    vcell('inspect', H5AD_TO_INSPECT)""")
md("## 8 建立并编辑数据配置\n运行后在左侧文件浏览器打开输出的 YAML，改四个 path 以及实际 gene、perturbation、batch、counts、control 字段。该格只创建一次模板。")
code("""import yaml
REAL_MANIFEST = WORK / 'configs/data_real.yaml'
REAL_DATA = WORK / 'prepared_real'
if not REAL_MANIFEST.exists():
    data_cfg = yaml.safe_load((PROJECT / 'configs/data.example.yaml').read_text())
    data_cfg['output_dir'] = str(REAL_DATA)
    for entry in data_cfg['datasets']:
        entry['path'] = str(WORK / 'raw' / Path(entry['path']).name)
    REAL_MANIFEST.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True))
print('编辑此文件:', REAL_MANIFEST)
print(REAL_MANIFEST.read_text())""")
md("## 9 预处理真实数据\n字段核对后再开启。运行完成必须看到 train、val、test 都有有效组。重新预处理请用新的 output_dir。")
code("""PREPARE_REAL = False
if PREPARE_REAL:
    vcell('prepare', '--manifest', REAL_MANIFEST)
    display(json.loads((REAL_DATA / 'data_audit.json').read_text()))
    real_meta = pd.read_csv(REAL_DATA / 'metadata.csv')
    display(real_meta.groupby(['split', 'context']).agg(
        groups=('row_id', 'count'), targets=('perturbation', 'nunique')))""")
md("""## 10 真实数据试跑
首轮使用 1 个 seed、20 epoch，五组实验均保留。完成后先读报告。
确认实验可将 SEEDS 改为 [0, 1, 2]，teacher/student epoch 上限改为 100/80，patience 改为 15；
同时修改 RUN_NAME，避免覆盖首轮结果。配置写入 WORK，便于恢复和记录。""")
code("""RUN_REAL = False
RUN_NAME = 'real_pilot_01'
SEEDS = [0]
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 20
PATIENCE = 6
REAL_RUN = WORK / RUN_NAME
TRAIN_CONFIG = WORK / 'configs' / (RUN_NAME + '.yaml')
if RUN_REAL:
    cfg = yaml.safe_load((PROJECT / 'configs/real.yaml').read_text())
    cfg.update(data_dir=str(REAL_DATA), output_dir=str(REAL_RUN), seeds=SEEDS,
               teacher_epochs=TEACHER_EPOCHS, student_epochs=STUDENT_EPOCHS,
               patience=PATIENCE)
    TRAIN_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = ['run', '--config', TRAIN_CONFIG]
    if (REAL_RUN / 'run_manifest.json').exists():
        args.append('--resume')
    vcell(*args)
    ACTIVE_RUN = REAL_RUN
    display(pd.read_csv(ACTIVE_RUN / 'evaluation_validation/summary.csv'))
    display(pd.read_csv(ACTIVE_RUN / 'evaluation_validation/paired_comparisons.csv'))""")
md("## 11 固定方案后评价测试集\n主比较先固定为 contrastive/mean 对 kd/mean。不要根据测试集排名继续调参。")
code("""REVEAL_TEST = False
if REVEAL_TEST:
    vcell('evaluate', '--run', ACTIVE_RUN, '--include-test')
    display(pd.read_csv(ACTIVE_RUN / 'evaluation_with_test/paired_comparisons.csv'))""")
md("## 12 新背景仅用对照推理\n按验证结果先确定 SELECTED_MODE，再填写 query YAML 和 targets.csv。checkpoint 已保存学生混合权重。")
code("""RUN_PREDICTION = False
SELECTED_MODE = 'supervised'  # 按验证结果确定后固定
QUERY_MANIFEST = ''
TARGETS_CSV = ''
if RUN_PREDICTION:
    checkpoint = ACTIVE_RUN / 'seed_0' / SELECTED_MODE / 'best.pt'
    vcell('predict-controls', '--checkpoint', checkpoint, '--manifest', QUERY_MANIFEST,
          '--targets', TARGETS_CSV, '--output', WORK / 'new_context_01')""")
md("## 13 下载报告\nDrive 已保存完整运行目录。此处可额外下载验证报告 ZIP；权重仍在 WORK 的运行目录中。")
code("""DOWNLOAD_REPORT = False
if DOWNLOAD_REPORT:
    import shutil
    report_zip = shutil.make_archive(str(WORK / 'validation_report'), 'zip',
                                     ACTIVE_RUN / 'evaluation_validation')
    files.download(report_zip)""")
md("""## 14 推送源码到 GitHub
先创建自己的空仓库，不预建 README。填写 URL、姓名和邮箱。
Fine-grained token 仅选择该仓库，Contents 设 Read and write；本包含 .github/workflows，
推送该文件还需 Workflows 的写权限。token 通过隐藏输入填写。
源码推送不包括 WORK 中的训练配置、数据和权重；这些仍应保存在 Drive。""")
code("""DO_PUSH = False
GITHUB_REPO = 'https://github.com/YOUR_USERNAME/vcell-teacher-student.git'
GIT_NAME = ''
GIT_EMAIL = ''
if DO_PUSH:
    import getpass, importlib.util
    spec = importlib.util.spec_from_file_location('source_push', PROJECT / 'scripts/push_github.py')
    source_push = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_push)
    token = getpass.getpass('GitHub token（隐藏）: ')
    try:
        source_push.push(GITHUB_REPO, GIT_NAME, GIT_EMAIL, token, confirm=True)
    finally:
        del token""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"name": "VCell_TeacherStudent_Colab.ipynb"}}
nbf.validate(nb)
nbf.write(nb, ROOT / "notebooks/VCell_TeacherStudent_Colab.ipynb")

