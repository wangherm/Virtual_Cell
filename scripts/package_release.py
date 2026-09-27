"""Build a source-only release; optionally attach an explicitly labelled synthetic QA report."""
from pathlib import Path
import argparse
import hashlib
import shutil
import zipfile


def build(output, qa_run=None):
    root = Path(__file__).resolve().parents[1]
    if qa_run:
        qa = root / "docs/qa"
        qa.mkdir(exist_ok=True)
        run = Path(qa_run) / "evaluation_validation"
        shutil.copy2(run / "report.html", qa / "synthetic_validation_report.html")
        shutil.copy2(run / "summary.csv", qa / "synthetic_summary.csv")
        shutil.copy2(run / "paired_comparisons.csv", qa / "synthetic_paired_comparisons.csv")
    allowed = ["src", "configs", "tests", "optional_tests", "scripts", "notebooks", "docs", ".github",
               "README.md", "pyproject.toml", ".gitignore", "LICENSE"]
    files = []
    for name in allowed:
        path = root / name
        files.extend([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()])
    files = [p for p in files if "__pycache__" not in p.parts and not p.suffix == ".pyc"
             and not any(x.endswith(".egg-info") for x in p.parts)]
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path("vcell_dual") / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP integrity test failed")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"{output}\n{len(files)} files; {output.stat().st_size} bytes\nSHA256 {checksum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--qa-run")
    args = parser.parse_args()
    build(args.output, args.qa_run)
