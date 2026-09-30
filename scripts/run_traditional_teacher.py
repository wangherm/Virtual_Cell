"""Native CellOracle/scTenifoldKnk review workers; no surrogate algorithms."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import urllib.request

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")


def human_prior(path):
    """Cache the official prior on the data disk, with atomic download and validation."""
    if not path.exists():
        url = "https://raw.githubusercontent.com/morris-lab/CellOracle/master/celloracle/data/promoter_base_GRN/hg38_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet"
        partial = path.with_suffix(".part")
        import time
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
                    import shutil
                    shutil.copyfileobj(response, target)
                pd.read_parquet(partial)  # Never accept an HTML error response as a prior.
                partial.replace(path)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    return pd.read_parquet(path)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_sctenifold(raw, symbols, targets, output, seed, threads, max_genes):
    from scTenifold import scTenifoldKnk
    # A reduced network makes this a bounded mechanism diagnostic, not a full-panel predictor.
    targets = [p for p in targets if p in set(symbols)]
    must = [i for i, s in enumerate(symbols) if s in targets]
    remaining = [i for i in np.argsort(-raw.var(0), kind="stable") if i not in set(must)]
    cols = np.array((must + remaining)[:max(max_genes, len(must))])
    frame = pd.DataFrame(raw[:, cols].T, index=symbols[cols])
    model = scTenifoldKnk(frame, ko_genes=[], qc_kws={"min_lib_size": 0,
        "remove_outlier_cells": False, "min_percent": 0, "min_exp_avg": 0, "min_exp_sum": 0,
        "max_mito_ratio": 1., "plot": False}, nc_kws={"n_nets": 3, "n_samp_cells": min(300, len(raw)),
        "backend": "joblib-threading", "n_jobs": threads, "random_state": seed})
    for stage in ("qc", "nc", "td"):
        model.run_step(stage)
    records = []
    for target in targets:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", target):
            raise ValueError("Unsafe target filename")
        if target not in model.tensor_dict["WT"].index:
            records.append({"target": target, "status": "unsupported_after_qc"})
            continue
        if np.abs(model.tensor_dict["WT"].loc[target].to_numpy()).sum() <= 1e-12:
            records.append({"target": target, "status": "no_outgoing_edges"})
            continue
        model.run_step("ko", ko_genes=[target])
        model.run_step("ma")
        model.run_step("dr")
        result = model.d_regulation
        if not {"Gene", "Distance"} <= set(result) or not np.isfinite(result.Distance).all():
            raise ValueError("Unexpected scTenifoldKnk native output")
        result.to_csv(output / f"{target}.csv", index=False)
        records.append({"target": target, "status": "complete", "genes": len(result)})
        print(f"SCTENIFOLD target={target} genes={len(result)}", flush=True)
    return records, {"prediction_space": "unsigned regulatory distance; NOT delta expression",
                     "network_genes": model.tensor_dict["WT"].index.tolist()}


def run_celloracle(raw, library, symbols, targets, output, seed, threads, prior_path=None):
    import anndata as ad
    import celloracle as co
    from sklearn.decomposition import PCA
    # Full-library denominators were captured before restricting the input gene panel.
    x = np.log1p(raw * (10000. / library)[:, None]).astype(np.float32)
    prior = pd.read_parquet(prior_path) if prior_path else co.data.load_human_promoter_base_GRN(version="hg38_gimmemotifsv5_fpr2")
    prior_file = output.parent / "human_promoter_prior.parquet"
    if not prior_file.exists():
        prior.to_parquet(prior_file)
    a = ad.AnnData(x, obs=pd.DataFrame(index=[f"control_{i}" for i in range(len(x))]),
                   var=pd.DataFrame(index=symbols))
    a.obs["background"] = pd.Categorical(["control"] * len(a))
    a.layers["raw_count"] = raw.copy()
    a.uns["background_colors"] = ["#4C78A8"]
    a.obsm["X_pca"] = PCA(n_components=2, random_state=seed).fit_transform(x)
    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(a, cluster_column_name="background", embedding_name="X_pca")
    oracle.import_TF_data(TF_info_matrix=prior)
    oracle.perform_PCA()
    oracle.knn_imputation(n_pca_dims=min(20, len(x)-1, x.shape[1]-1), k=min(15, len(x)-1), n_jobs=threads)
    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10, use_cluster_specific_TFdict=False)
    records = []
    for target in targets:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", target):
            raise ValueError("Unsafe target filename")
        if target not in set(oracle.active_regulatory_genes):
            records.append({"target": target, "status": "unsupported_not_active_regulator"})
            continue
        try:
            oracle.simulate_shift(perturb_condition={target: 0.}, GRN_unit="whole", n_propagation=3)
        except ValueError as error:
            if "perturbation condition is far" not in str(error):
                raise
            records.append({"target": target, "status": "rejected_out_of_expression_range", "reason": str(error)})
            continue
        delta = np.asarray(oracle.adata.layers["delta_X"]).mean(0)
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite CellOracle simulation")
        pd.DataFrame({"Gene": oracle.adata.var_names, "delta": delta}).to_csv(output / f"{target}.csv", index=False)
        records.append({"target": target, "status": "complete", "genes": len(delta)})
        print(f"CELLORACLE target={target} genes={len(delta)}", flush=True)
    return records, {"prediction_space": "mean simulated minus imputed control in log1p full-library-normalized space",
        "caveat": "Pooled control-cell simulation; KO=0 is not experimentally measured CRISPRi efficiency",
        "prior_sha256": hashlib.sha256(prior_file.read_bytes()).hexdigest(), "prior": "hg38_gimmemotifsv5_fpr2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["celloracle", "sctenifold"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--network-genes", type=int, default=512)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("celloracle" if args.family == "celloracle" else "scTenifoldpy")
    expected = "0.20.0" if args.family == "celloracle" else "0.4.0"
    if version != expected:
        raise ValueError(f"Require {args.family} {expected}, got {version}")
    facts = json.loads((args.inputs / "audit.json").read_text())
    all_records = []
    for entry in facts["datasets"]:
        context = entry["context"]
        folder = args.output / context
        folder.mkdir(exist_ok=True)
        if (folder / "COMPLETE.json").exists():
            all_records.append(json.loads((folder / "COMPLETE.json").read_text()))
            continue
        with np.load(args.inputs / entry["file"], allow_pickle=False) as f:
            raw, symbols, library = f["raw"], f["symbols"], f["library"]
        random.seed(17)
        np.random.seed(17)
        targets = entry["targets"]
        # CellOracle targets must be candidate TFs; selection uses a prior, never outcomes.
        if args.family == "celloracle":
            import celloracle as co
            prior_file = args.prior or args.output / "human_promoter_prior.parquet"
            prior = human_prior(prior_file)
            candidates = [t for t in targets if t in prior.columns]
            selected = candidates[:args.max_targets]
            records, detail = run_celloracle(raw, library, symbols, selected, folder, 17, args.threads, prior_file)
        else:
            selected = targets[:args.max_targets]
            records, detail = run_sctenifold(raw, symbols, selected, folder, 17, args.threads, args.network_genes)
        report = {"family": args.family, "version": version, "context": context,
            "available_targets": targets, "selected_targets": selected, "records": records, **detail,
            "test_evaluated": False, "source_controls_sha256": entry["sha256"]}
        write(folder / "COMPLETE.json", report)
        all_records.append(report)
    done = sum(r["status"] == "complete" for c in all_records for r in c["records"])
    if not done:
        write(args.output / "FAILED.json", {"reason": "No supported target produced a prediction", "contexts": all_records})
        raise RuntimeError("No supported traditional-teacher predictions")
    write(args.output / "COMPLETE.json", {"family": args.family, "version": version,
        "contexts": all_records, "completed_target_contexts": done, "test_evaluated": False})


if __name__ == "__main__":
    main()
