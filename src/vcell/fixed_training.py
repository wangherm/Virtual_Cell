"""Fixed-update Qwen experiments, explicit exposure counters and signed response bases."""
import copy
import gc
import hashlib
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd

from .selection import mse, row_weights
from .train import normalization, tensors
from .utils import atomic_torch_save, seed_all, write_json


def response_basis(data, fit, rank):
    """One mean per context/target: repeated rows do not dominate SVD fitting."""
    meta = data["meta"].iloc[fit].reset_index(drop=True)
    groups = meta.groupby(["context", "perturbation"], sort=True).indices
    matrix = np.stack([data["delta"][fit[ix]].mean(0) for ix in groups.values()])
    _, singular, vt = svd(matrix, full_matrices=False, check_finite=True)
    count = min(int(rank), len(vt))
    basis = vt[:count].astype(np.float32)
    for v in basis:
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
    return basis, {"requested_rank": rank, "actual_rank": count,
        "fit_rows": fit.tolist(), "context_target_units": len(groups),
        "energy_fraction": float((singular[:count]**2).sum() / max((singular**2).sum(), 1e-30)),
        "centering": "none; signed delta relative to zero", "test_evaluated": False}


def step_rows(train, seed, step, batch):
    """A deterministic stream of complete shuffled passes; full batches across boundaries."""
    start, left, chunks = step * batch, batch, []
    while left:
        epoch, offset = divmod(start, len(train))
        order = np.random.default_rng(seed + epoch).permutation(train)
        take = min(left, len(train) - offset)
        chunks.append(order[offset:offset+take])
        start, left = start + take, left - take
    return np.concatenate(chunks)


def train_fixed_arm(cfg, data, targets, arm, train, val, dest, stamp, resume):
    from .qwen import QwenResponse, backbone_identity, predict, training_weights
    if arm != "supervised":
        raise ValueError("Round 7 fixed-step prototype currently uses genuine supervised labels only")
    max_steps, cadence = int(cfg["max_steps"]), int(cfg["eval_every_steps"])
    if min(max_steps, cadence) < 1:
        raise ValueError("Fixed step counts must be positive")
    local = copy.copy(data)
    local["splits"] = {**data["splits"], "train": train}
    norm = normalization(local)
    seed_all(cfg["seed"])
    basis, basis_audit = (None, None)
    if cfg["model"].get("response_rank"):
        basis, basis_audit = response_basis(data, train, cfg["model"]["response_rank"])
    reference = None
    if cfg["model"].get("background_interaction"):
        from .specialization import reference_responses
        reference = reference_responses(data, train)[0] / norm["scale"]
    device = torch.device(cfg["device"])
    model = QwenResponse(len(data["genes"]), data["perturbations"], cfg["model"],
                         reference_values=reference, response_basis=basis).to(device)
    folder = dest / arm
    folder.mkdir(exist_ok=True)
    if basis_audit:
        write_json(folder / "basis_audit.json", basis_audit)
    params = [p for p in model.parameters() if p.requires_grad]
    shared = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad and not name.startswith(("output.", "background_")):
            shared.update(name.encode())
            shared.update(p.detach().float().cpu().numpy().tobytes())
    write_json(folder / "model_info.json", {"total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in params), "shared_initial_sha256": shared.hexdigest()})
    common = {"format": "vcell-qwen-adapter-v1", "model_spec": cfg["model"],
        "backbone_identity": backbone_identity(cfg["model"]), "genes": data["genes"].tolist(),
        "perturbations": data["perturbations"].tolist(), "prompt_ids": model.prompt_ids.cpu(),
        "prompt_mask": model.prompt_mask.cpu(), "response_basis": None if basis is None else torch.tensor(basis),
        "reference_values": None if reference is None else torch.tensor(reference),
        "normalization": {"mean": torch.tensor(norm["mean"]), "std": torch.tensor(norm["std"]), "scale": norm["scale"]},
        "data_fingerprint": data["audit"]["fingerprint"], "run_fingerprint": stamp}
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    ts = tensors(data, norm)
    weights, _ = training_weights(cfg, data, train)
    batch = cfg["batch_size"] * cfg["gradient_accumulation"]
    first, history, best, exposure = 0, [], float("inf"), {}
    last = folder / "last.pt"
    if resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["run_fingerprint"] != stamp:
            raise ValueError("Fixed-step checkpoint mismatch")
        model.load_adapter_state(saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer"])
        first, history, best, exposure = saved["optimizer_step"], saved["history"], saved["best"], saved["exposure"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timer = time.monotonic()
    accumulated, counted = 0., 0
    for step in range(first, max_steps):
        seed_all(cfg["seed"] + 10000 + step)
        ix = step_rows(train, cfg["seed"], step, batch)
        # An explicit constant step-based schedule preserves the earlier LR setting.
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for offset in range(0, batch, cfg["batch_size"]):
            rows = ix[offset:offset+cfg["batch_size"]]
            prediction = model(ts["x"][rows].to(device), ts["p"][rows].to(device))
            loss = ((prediction-ts["y"][rows].to(device)).square().mean(1) * weights[rows].to(device)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fixed-step loss")
            (loss * len(rows) / batch).backward()
            loss_value += loss.item() * len(rows) / batch
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        for context, count in data["meta"].iloc[ix].context.value_counts().items():
            exposure[context] = exposure.get(context, 0) + int(count)
        accumulated, counted = accumulated + loss_value, counted + 1
        done = step + 1
        if done == 1 or done % cfg["log_every"] == 0:
            print(f"FIXED step={done}/{max_steps} examples={done*batch} loss={loss_value:.6f}", flush=True)
        if done % cadence == 0 or done == max_steps:
            pred = predict(model, ts, val, device, cfg["batch_size"]) * norm["scale"]
            score = mse(pred, data["delta"][val], row_weights(data["meta"].iloc[val]))
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite fixed-step validation")
            state = {**common, "trainable_state": model.adapter_state(), "optimizer_step": done, "epoch": done*batch/len(train)}
            if score < best:
                best = score
                atomic_torch_save(state, folder / "best.pt")
            np.savez_compressed(folder / f"curve_step{done}.npz", delta=pred,
                                row_ids=data["meta"].iloc[val].row_id.to_numpy(dtype="U"), genes=data["genes"])
            record = {"optimizer_step": done, "examples_drawn": done*batch, "learning_rate": cfg["learning_rate"],
                "loss": accumulated / max(counted, 1), "val_mse_delta": score,
                "interval_seconds": time.monotonic()-timer, **{"examples_"+k: v for k,v in exposure.items()}}
            history.append(record)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            atomic_torch_save({**state, "optimizer": optimizer.state_dict(), "history": history,
                              "best": best, "exposure": exposure}, last)
            print(f"FIXED EVAL step={done} val_mse_delta={score:.7f} exposure={exposure}", flush=True)
            timer, accumulated, counted = time.monotonic(), 0., 0
    # Primary result ALWAYS comes from the predeclared final update, never best.pt.
    atomic_torch_save({**common, "trainable_state": model.adapter_state(), "optimizer_step": max_steps,
                      "epoch": max_steps*batch/len(train)}, folder / "endpoint.pt")
    pred = predict(model, ts, val, device, cfg["batch_size"]) * norm["scale"]
    np.savez_compressed(folder / "fixed_predictions.npz", delta=pred, genes=data["genes"],
        row_ids=data["meta"].iloc[val].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
    if cfg.get("internal_split"):
        cal = np.asarray(cfg["internal_split"]["calibration"], dtype=int)
        cp = predict(model, ts, cal, device, cfg["batch_size"]) * norm["scale"]
        np.savez_compressed(folder / "calibration_predictions.npz", delta=cp, genes=data["genes"],
            row_ids=data["meta"].iloc[cal].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
    write_json(folder / "fixed_endpoint.json", {"optimizer_step": max_steps, "examples_drawn": max_steps*batch,
        "examples_by_context": exposure, "mse_delta": mse(pred, data["delta"][val], row_weights(data["meta"].iloc[val])),
        "validation_best_supplementary": best, "test_evaluated": False,
        "normalization_fit_rows": train.tolist(), "target_scale": norm["scale"],
        "objective": "row-uniform normalized-delta MSE" if cfg.get("loss_weighting", "uniform") == "uniform" else "context-target weighted normalized-delta MSE",
        "schedule": "constant LR per optimizer update; no early stopping",
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    del model, optimizer, params
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred
