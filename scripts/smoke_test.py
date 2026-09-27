"""Run the complete synthetic workflow from a terminal, without a notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("runs/smoke-check"),
                        help="Data and run directory; relative to the current directory")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse this smoke test's prepared data and resume the same run")
    args = parser.parse_args(argv)
    work = args.work_dir.resolve()
    demo, run = work / "demo", work / "run"
    prepared = demo / "prepared"

    if args.resume:
        required = [prepared / name for name in
                    ("dataset.npz", "metadata.csv", "data_audit.json")]
        if not all(p.is_file() for p in required):
            parser.error("Cannot resume without complete prepared data; use a new --work-dir.")
        if not json.loads(required[-1].read_text(encoding="utf-8")).get("synthetic"):
            parser.error("The smoke workflow only accepts its synthetic prepared data.")
    elif work.exists() and (not work.is_dir() or any(work.iterdir())):
        parser.error("Work directory is not empty; choose a new --work-dir or use --resume.")

    def vcell(*command):
        subprocess.run([sys.executable, "-u", "-m", "vcell", *map(str, command)],
                       cwd=ROOT, check=True)

    if not args.resume:
        vcell("demo", "--output", demo)
    command = ["run", "--config", ROOT / "configs/smoke.yaml", "--data", prepared,
               "--output", run, "--device", args.device]
    if args.resume:
        command.append("--resume")
    vcell(*command)
    report = run / "evaluation_validation/report.html"
    if not report.is_file():
        raise RuntimeError(f"Training did not produce the expected report: {report}")
    print(f"\nSmoke test complete. Validation report: {report}")
    print(f"Checkpoints and histories: {run / 'seed_0'}")
    print("Synthetic results validate software execution, not biological performance.")


if __name__ == "__main__":
    main()
