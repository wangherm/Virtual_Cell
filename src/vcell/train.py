from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .data import load_prepared
from .models import make_model
from .losses import student_losses
from .selection import row_weights, mse, combine, summarize
from .teachers import load_cache
from .utils import seed_all, atomic_torch_save, device_from, write_json, file_sha256

MODES = {"supervised": (0, 0, 0), "kd": (1, 0, 0), "mutual": (1, 1, 0),
         "contrastive": (1, 0, 1), "mutual_contrastive": (1, 1, 1)}
TUNABLE = ("learning_rate", "weight_decay", "kd_weight", "peer_weight", "contrast_weight")


def normalization(data):
    t = data["splits"]["train"]
    return {"mean": data["baseline"][t].mean(0).astype(np.float32),
            "std": np.maximum(data["baseline"][t].std(0), .1).astype(np.float32),
            "scale": max(float(np.sqrt(np.mean(data["delta"][t] ** 2))), .05)}


def tensors(data, norm):
    return {"x": torch.tensor((data["baseline"] - norm["mean"]) / norm["std"]),
            "p": torch.tensor(data["pert_idx"], dtype=torch.long),
            "y": torch.tensor(data["delta"] / norm["scale"])}


@torch.no_grad()
def predict_models(models, ts, indices, device, batch_size=128):
    for model in models:
        model.eval()
    output = [[] for _ in models]
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        x, p = ts["x"][idx].to(device), ts["p"][idx].to(device)
        for i, model in enumerate(models):
            output[i].append(model(x, p)[0].cpu().numpy())
    return [np.concatenate(x) for x in output]


def _cpu_states(models):
    return [{k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for model in models]


def fit_models(specs, data, cfg, out, seed, epochs, settings, teacher_pred=None, resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = device_from(cfg["device"])
    norm = normalization(data)
    ts = tensors(data, norm)
    models, saved_specs = [], []
    for i, spec in enumerate(specs):
        seed_all(seed + i * 1009)
        model, kwargs = make_model(spec, len(data["genes"]), len(data["perturbations"]), cfg["projection_dim"])
        models.append(model.to(device))
        saved_specs.append({"name": spec["name"], "architecture": spec["architecture"], "kwargs": kwargs})
    parameters = [p for m in models for p in m.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best_score, best_states, stale, first, history = float("inf"), None, 0, 0, []
    last_path = out / "last.pt"
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        for m, state in zip(models, saved["states"]):
            m.load_state_dict(state)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        best_score, best_states = saved["best_score"], saved["best_states"]
        stale, first, history = saved["stale"], saved["epoch"] + 1, saved["history"]
        if saved["finished"]:
            first = epochs
    target = None
    if settings["kd_weight"]:
        target = torch.tensor(combine(teacher_pred, settings["teacher_weights"]) / norm["scale"])
    train_idx, val_idx = data["splits"]["train"], data["splits"]["val"]
    val_weights = row_weights(data["meta"].iloc[val_idx])
    for epoch in range(first, epochs):
        started = time.time()
        seed_all(seed + 100000 + epoch)
        order = np.random.default_rng(seed + epoch).permutation(train_idx)
        for m in models:
            m.train()
        stats = {"supervised": 0., "kd": 0., "peer": 0., "contrastive": 0., "loss": 0.}
        ramp = 0. if epoch < cfg["warmup_epochs"] else min(1., (epoch - cfg["warmup_epochs"] + 1) / 5)
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            x, p, y = (ts[k][idx].to(device) for k in ("x", "p", "y"))
            optimizer.zero_grad(set_to_none=True)
            outputs = [m(x, p) for m in models]
            sup = torch.stack([F.mse_loss(o[0], y) for o in outputs]).mean()
            kd = torch.stack([F.mse_loss(o[0], target[idx].to(device)) for o in outputs]).mean() if target is not None else sup * 0
            peer, contrast = student_losses(outputs, p, cfg["temperature"],
                                             bool(settings["peer_weight"]), bool(settings["contrast_weight"]))
            loss = sup + settings["kd_weight"] * kd + ramp * (settings["peer_weight"] * peer + settings["contrast_weight"] * contrast)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            for key, value in zip(stats, (sup, kd, peer, contrast, loss)):
                stats[key] += float(value.detach().cpu()) * len(idx)
        val_pred = predict_models(models, ts, val_idx, device, cfg["batch_size"])
        truth = ts["y"][val_idx].numpy()
        val_mse = mse(np.mean(val_pred, axis=0), truth, val_weights)
        if val_mse < best_score - 1e-8:
            best_score, best_states, stale = val_mse, _cpu_states(models), 0
        else:
            stale += 1
        record = {"epoch": epoch + 1, **{k: v / len(order) for k, v in stats.items()},
                  "val_normalized_mse": val_mse, "ramp": ramp, "seconds": time.time() - started}
        record.update({f"val_{s['name']}": mse(p, truth, val_weights) for s, p in zip(specs, val_pred)})
        history.append(record)
        finished = stale >= int(cfg["patience"]) or epoch + 1 == epochs
        atomic_torch_save({"states": _cpu_states(models), "optimizer": optimizer.state_dict(),
                           "best_states": best_states, "best_score": best_score, "stale": stale,
                           "epoch": epoch, "history": history, "finished": finished}, last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"{out.name} epoch={epoch+1} loss={record['loss']:.4f} val={val_mse:.4f}", flush=True)
        if finished:
            break
    for m, state in zip(models, best_states):
        m.load_state_dict(state)
    raw = predict_models(models, ts, np.arange(len(data["meta"])), device)
    predictions = {s["name"]: p * norm["scale"] for s, p in zip(specs, raw)}
    validation = summarize({k: p[val_idx] for k, p in predictions.items()},
                           data["delta"][val_idx], data["meta"].iloc[val_idx])
    validation.update(epochs_run=len(history), best_epoch=min(history, key=lambda r: r["val_normalized_mse"])["epoch"],
                      seconds=sum(r["seconds"] for r in history))
    write_json(out / "validation.json", validation)
    bundle = {"specs": saved_specs, "states": best_states,
              "ensemble_weights": list(validation["weights"].values()),
              "normalization": {"mean": torch.tensor(norm["mean"]), "std": torch.tensor(norm["std"]), "scale": norm["scale"]},
              "genes": data["genes"].tolist(), "perturbations": data["perturbations"].tolist(),
              "target_sum": float(data["audit"]["target_sum"]), "seed": seed,
              "data_fingerprint": data["audit"]["fingerprint"], "settings": settings,
              "parameter_counts": [sum(p.numel() for p in m.parameters()) for m in models]}
    atomic_torch_save(bundle, out / "best.pt")
    write_json(out / "model_info.json", {"specs": saved_specs, "parameter_counts": bundle["parameter_counts"]})
    return predictions


def simple_baselines(data):
    train = data["splits"]["train"]
    by_target = {}
    for p in np.unique(data["pert_idx"][train]):
        ix = train[data["pert_idx"][train] == p]
        means = [data["delta"][ix[data["meta"].iloc[ix].context.to_numpy() == c]].mean(0)
                 for c in data["meta"].iloc[ix].context.unique()]
        by_target[int(p)] = np.mean(means, axis=0)
    return {"no_change": np.zeros_like(data["delta"]),
            "mean_transfer": np.stack([by_target[int(p)] for p in data["pert_idx"]])}


def settings_for(cfg, mode):
    settings = {k: cfg[k] for k in TUNABLE}
    for k, m in zip(("kd_weight", "peer_weight", "contrast_weight"), MODES[mode]):
        settings[k] *= m
    settings["teacher_weights"] = [1. / len(cfg["teachers"])] * len(cfg["teachers"])
    return settings


def ensemble_predictions(name, predictions, data):
    v = data["splits"]["val"]
    stats = summarize({k: p[v] for k, p in predictions.items()}, data["delta"][v], data["meta"].iloc[v])
    output = {f"{name}/{k}": p for k, p in predictions.items()}
    output[f"{name}/mean"] = np.mean(list(predictions.values()), axis=0)
    output[f"{name}/valmix"] = combine(list(predictions.values()), list(stats["weights"].values()))
    return output, stats


def save_predictions(folder, predictions, data):
    path = Path(folder) / "predictions.npz"
    existing = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            existing = {k: f[k] for k in f.files if k != "data_fingerprint"}
    existing.update(predictions)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **existing, data_fingerprint=data["audit"]["fingerprint"])
    tmp.replace(path)


def source_fingerprint():
    h = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        h.update(source.name.encode())
        h.update(source.read_bytes())
    return h.hexdigest()


def load_run(run_dir):
    manifest = json.loads((Path(run_dir) / "run_manifest.json").read_text())
    data = load_prepared(manifest["config"]["data_dir"])
    if data["audit"]["fingerprint"] != manifest["data_fingerprint"]:
        raise ValueError("Run/data fingerprint mismatch")
    return manifest, data


def run_trial(run_dir, name, settings, data, cfg, resume=False):
    for seed in cfg["seeds"]:
        folder = Path(run_dir) / f"seed_{seed}"
        with np.load(folder / "predictions.npz", allow_pickle=False) as f:
            teachers = [f[f"teachers/{s['name']}"] for s in cfg["teachers"]]
        pred = fit_models(cfg["students"], data, cfg, folder / name, int(seed),
                          cfg["student_epochs"], settings, teachers, resume)
        output, _ = ensemble_predictions(name, pred, data)
        save_predictions(folder, output, data)


def run_suite(cfg, resume=False):
    if not cfg["experiments"] or not set(cfg["experiments"]).issubset(MODES):
        raise ValueError(f"Choose experiments from {list(MODES)}")
    if not cfg["teachers"] or not cfg["students"]:
        raise ValueError("Configure at least one teacher and student")
    for role in ("teachers", "students"):
        names = [s["name"] for s in cfg[role]]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", n) or n in {"mean", "valmix"} for n in names):
            raise ValueError(f"Use unique simple names in {role}; mean and valmix are reserved")
    if min(cfg["teacher_epochs"], cfg["student_epochs"], cfg["batch_size"], cfg["patience"]) < 1:
        raise ValueError("Epochs, batch size and patience must be positive")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Use nonempty, unique seeds")
    torch.set_num_threads(int(cfg["num_threads"]))
    data = load_prepared(cfg["data_dir"])
    dest = Path(cfg["output_dir"])
    cache_hashes = {s["name"]: [file_sha256(s["cache"]), file_sha256(Path(s["cache"]).with_suffix(".json"))]
                    for s in cfg["teachers"] if "cache" in s}
    stamp = {"config": cfg, "data_fingerprint": data["audit"]["fingerprint"],
             "source_fingerprint": source_fingerprint(), "external_cache_hashes": cache_hashes}
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    if dest.exists() and any(dest.iterdir()):
        if not resume or not (dest / "run_manifest.json").exists():
            raise FileExistsError(f"{dest} exists; choose a new output or --resume")
        old = json.loads((dest / "run_manifest.json").read_text())
        if old["run_fingerprint"] != fingerprint:
            raise ValueError("Configuration, source, cache or data changed; choose a new run")
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "run_manifest.json", {**stamp, "run_fingerprint": fingerprint,
              "synthetic": data["audit"]["synthetic"], "python": sys.version,
              "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available()})
    for seed in cfg["seeds"]:
        folder = dest / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        teachers, sources = {}, {}
        for i, spec in enumerate(cfg["teachers"]):
            if "cache" in spec:
                teachers[spec["name"]], sources[spec["name"]] = load_cache(spec["cache"], data)
            else:
                teachers.update(fit_models([spec], data, cfg, folder / "teachers" / spec["name"],
                                           int(seed) + 2000 + i * 100, cfg["teacher_epochs"],
                                           settings_for(cfg, "supervised"), resume=resume))
        output, stats = ensemble_predictions("teachers", teachers, data)
        write_json(folder / "teacher_validation.json", stats)
        write_json(folder / "teacher_sources.json", sources)
        v = data["splits"]["val"]
        rw = row_weights(data["meta"].iloc[v])
        write_json(folder / "baseline_validation.json", {k: mse(p[v], data["delta"][v], rw)
                   for k, p in simple_baselines(data).items()})
        save_predictions(folder, {**simple_baselines(data), **output}, data)
    for mode in cfg["experiments"]:
        run_trial(dest, mode, settings_for(cfg, mode), data, cfg, resume)
    from .evaluate import evaluate_run
    evaluate_run(dest)
    return dest
