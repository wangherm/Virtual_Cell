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


if __name__ == "__main__":
    main()
