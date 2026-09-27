from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .models import MODEL_TYPES
from .data import aggregate_source
from .utils import read_config, device_from
from .selection import combine


def load_bundle(path, device="cpu"):
    device = device_from(device)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for spec, state in zip(bundle["specs"], bundle["states"]):
        model = MODEL_TYPES[spec["architecture"]](**spec["kwargs"])
        model.load_state_dict(state)
        models.append(model.to(device).eval())
    return bundle, models, device


@torch.no_grad()
def predict_controls(checkpoint, query_manifest, targets_file, output, device="auto"):
    """New context inference from controls alone. Produces group means, not fake single cells."""
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(dest)
    bundle, models, device = load_bundle(checkpoint, device)
    manifest_path = Path(query_manifest).resolve()
    cfg = read_config(manifest_path)
    targets = pd.read_csv(targets_file, dtype=str)["perturbation"].tolist()
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Target CSV must have unique perturbation values")
    vocab = {p: i for i, p in enumerate(bundle["perturbations"])}
    if set(targets) - set(vocab):
        raise ValueError("Targets must be present in the training perturbation vocabulary")
    aggregation_cfg = {"chunk_size": cfg.get("chunk_size", 2048), "target_sum": bundle["target_sum"]}
    baselines, rows = [], []
    for entry in cfg["datasets"]:
        agg = aggregate_source(entry, manifest_path.parent, bundle["genes"], aggregation_cfg, set())
        for (batch, p), (summed, n) in sorted(agg.items()):
            if p != "__control__" or n < cfg.get("min_control_cells", 30):
                continue
            for target in targets:
                baselines.append(summed / n)
                rows.append({"dataset": entry["id"], "context": entry["context"],
                             "batch": batch, "perturbation": target, "n_controls": n})
    if not rows:
        raise ValueError("No sufficiently sampled control groups")
    base = np.asarray(baselines, dtype=np.float32)
    norm = bundle["normalization"]
    x = (torch.tensor(base) - norm["mean"]) / norm["std"]
    p = torch.tensor([vocab[r["perturbation"]] for r in rows])
    predictions = []
    for model in models:
        chunks = [model(x[s:s+128].to(device), p[s:s+128].to(device))[0].cpu().numpy()
                  for s in range(0, len(x), 128)]
        predictions.append(np.concatenate(chunks) * norm["scale"])
    mixed = combine(predictions, bundle["ensemble_weights"])
    dest.mkdir(parents=True)
    np.savez_compressed(dest / "mean_predictions.npz", delta=mixed, predicted_mean=base + mixed,
                        baseline=base, genes=np.asarray(bundle["genes"], dtype="U"),
                        **{f"model_{i}_delta": p for i, p in enumerate(predictions)})
    pd.DataFrame(rows).to_csv(dest / "prediction_metadata.csv", index=False)
    print(f"Saved {len(rows)} group means to {dest}. These are NOT single-cell samples.")
    return dest
