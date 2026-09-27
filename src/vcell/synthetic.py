"""Synthetic counts for software tests only. No biological efficacy claim."""
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import yaml
from .data import prepare
from .utils import write_json


def make_demo(dest, seed=17, genes=96, targets=20, cells=24):
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Use a new demo directory: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    factors = 6
    programs = rng.normal(0, .16, (factors, genes))
    target_factors = rng.normal(0, .8, (targets, factors))
    entries = []
    for c, context in enumerate(["K562", "RPE1", "HepG2", "Jurkat"]):
        state = rng.normal(0, .7, factors)
        base = 1.3 + state @ programs
        counts, obs = [], []
        for b in range(2):
            batch_offset = rng.normal(0, .03, genes)
            for p in range(-1, targets):
                effect = np.zeros(genes) if p < 0 else (target_factors[p] * (1 + .2 * state)) @ programs
                for j in range(cells * (2 if p < 0 else 1)):
                    rate = np.exp(base + batch_offset + effect + rng.normal(0, .12, genes))
                    counts.append(rng.poisson(rate))
                    obs.append({"target": "control" if p < 0 else f"PERT{p:03d}",
                                "batch": f"batch{b}"})
        a = ad.AnnData(sparse.csr_matrix(np.asarray(counts, dtype=np.int32)),
                       obs=pd.DataFrame(obs, index=[f"{context}_{i}" for i in range(len(obs))]),
                       var=pd.DataFrame(index=[f"GENE{i:04d}" for i in range(genes)]))
        a.write_h5ad(dest / f"{context}.h5ad")
        entries.append({"id": f"synthetic_{context}", "path": f"{context}.h5ad", "context": context,
                        "gene_key": None, "perturbation_key": "target", "batch_key": "batch",
                        "count_layer": "X", "control_values": ["control"]})
    cfg = {"output_dir": "prepared", "train_contexts": ["K562", "RPE1"], "val_contexts": ["HepG2"],
           "test_contexts": ["Jurkat"], "target_sum": 10000, "max_genes": genes,
           "max_perturbations": targets, "min_cells": 5, "min_control_cells": 10,
           "chunk_size": 256, "seed": seed, "datasets": entries}
    manifest = dest / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    prepared = prepare(manifest)
    import json
    audit = json.loads((prepared / "data_audit.json").read_text())
    audit["synthetic"] = True
    audit["warning"] = "Artificial counts. Results test software, NOT biological generalization."
    write_json(prepared / "data_audit.json", audit)
    return prepared
