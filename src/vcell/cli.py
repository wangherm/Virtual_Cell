from __future__ import annotations
import argparse
import json
from pathlib import Path
from .utils import read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="VCell: teacher-student perturbation experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Inspect actual h5ad field names before configuring")
    p.add_argument("path")
    p = sub.add_parser("demo", help="Generate artificial counts and run real preprocessing")
    p.add_argument("--output", default="data/demo")
    p.add_argument("--seed", type=int, default=17)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--include-test", action="store_true", help="Explicitly reveal held-out test results AFTER model selection")
    p = sub.add_parser("queries")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("import-teacher")
    for arg in ("data", "matrix", "rows", "genes", "provenance", "output"):
        p.add_argument("--" + arg, required=True)
    p = sub.add_parser("predict-controls")
    for arg in ("checkpoint", "manifest", "targets", "output"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p = sub.add_parser("catalogue", help="List author-released Figshare files; no bulk download")
    p.add_argument("--article", type=int, default=20029387)
    p = sub.add_parser("download")
    p.add_argument("--article", type=int, default=20029387)
    p.add_argument("--file-id", type=int, required=True)
    p.add_argument("--output", default="data/raw")
    p = sub.add_parser("qwen-run", help="Pretrained Qwen numerical regression and audited teacher distillation")
    p.add_argument("--config", required=True)
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model-dir", help="Existing local Qwen snapshot; avoids Hub metadata probes")
    p.add_argument("--offline", action="store_true", help="Forbid model/tokenizer downloads")
    p = sub.add_parser("qwen-predict", help="Reload a Qwen adapter and predict prepared control queries")
    for arg in ("checkpoint", "data", "output"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=16)
    for command in ("teacher-export", "teacher-fit-head"):
        p = sub.add_parser(command, help="Native pretrained teacher pipeline; see docs/FOUNDATION.md")
        p.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        from .data import inspect_h5ad
        print(json.dumps(inspect_h5ad(args.path), indent=2, ensure_ascii=False))
    elif args.command == "demo":
        from .synthetic import make_demo
        print(make_demo(args.output, args.seed))
    elif args.command == "prepare":
        from .data import prepare
        print(prepare(args.manifest))
    elif args.command == "run":
        from .train import run_suite
        cfg = read_config(args.config)
        if args.data:
            cfg["data_dir"] = args.data
        if args.output:
            cfg["output_dir"] = args.output
        if args.device:
            cfg["device"] = args.device
        cfg["data_dir"] = str(Path(cfg["data_dir"]).resolve())
        cfg["output_dir"] = str(Path(cfg["output_dir"]).resolve())
        for spec in cfg["teachers"]:
            if "cache" in spec:
                spec["cache"] = str(Path(spec["cache"]).resolve())
        print(run_suite(cfg, resume=args.resume))
    elif args.command == "evaluate":
        from .evaluate import evaluate_run
        print(evaluate_run(args.run, args.include_test))
    elif args.command == "queries":
        from .data import load_prepared
        from .teachers import export_queries
        export_queries(load_prepared(args.data), args.output)
    elif args.command == "import-teacher":
        from .data import load_prepared
        from .teachers import import_cache
        import_cache(args.matrix, args.rows, args.genes, args.provenance,
                     load_prepared(args.data), args.output)
    elif args.command == "predict-controls":
        from .inference import predict_controls
        predict_controls(args.checkpoint, args.manifest, args.targets, args.output, args.device)
    elif args.command == "catalogue":
        from .download import figshare_catalogue
        import pandas as pd
        files = figshare_catalogue(args.article)
        print(pd.DataFrame(files)[["id", "name", "size"]].to_string(index=False))
    elif args.command == "download":
        from .download import download_figshare
        download_figshare(args.file_id, args.output, args.article)
    elif args.command == "qwen-run":
        from .qwen import run_qwen
        cfg = read_config(args.config)
        if args.model_dir:
            model_dir = Path(args.model_dir).resolve()
            if not model_dir.is_dir():
                parser.error(f"Local model directory does not exist: {model_dir}")
            cfg["model"].update(model_id=str(model_dir), revision=None, local_files_only=True)
        if args.offline:
            cfg["model"]["local_files_only"] = True
        for key in ("data", "output"):
            value = getattr(args, key)
            if value:
                cfg[key + "_dir"] = value
            cfg[key + "_dir"] = str(Path(cfg[key + "_dir"]).resolve())
        cfg["teachers"] = {n: str(Path(p).resolve()) for n, p in cfg.get("teachers", {}).items()}
        print(run_qwen(cfg, resume=args.resume))
    elif args.command == "qwen-predict":
        from .qwen import predict_checkpoint
        predict_checkpoint(args.checkpoint, args.data, args.output, args.device, args.batch_size)
    elif args.command == "teacher-export":
        from .teacher_pipeline import export_native_teacher
        print(export_native_teacher(read_config(args.config)))
    elif args.command == "teacher-fit-head":
        from .teacher_pipeline import fit_feature_teacher
        print(fit_feature_teacher(read_config(args.config)))


if __name__ == "__main__":
    main()
