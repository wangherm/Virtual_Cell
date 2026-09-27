"""Mean-response diagnostics, not the official VCC single-cell score."""
from pathlib import Path
from itertools import combinations
import html
import json
import numpy as np
import pandas as pd
from .data import load_prepared
from .utils import write_json


def row_metrics(pred, truth, top_k=50):
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Need matching (groups, genes) matrices")
    error = pred - truth
    a, b = pred - pred.mean(1, keepdims=True), truth - truth.mean(1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    corr = np.divide((a * b).sum(1), denom, out=np.full(len(a), np.nan), where=denom > 1e-12)
    k = min(int(top_k), pred.shape[1])
    ti, pi = np.argsort(-np.abs(truth), axis=1)[:, :k], np.argsort(-np.abs(pred), axis=1)[:, :k]
    overlap, direction = [], []
    for i in range(len(pred)):
        # No arbitrary top-k score when all predictions/truth are tied at zero.
        if np.max(np.abs(pred[i])) <= 1e-12 or np.max(np.abs(truth[i])) <= 1e-12:
            overlap.append(np.nan)
        else:
            overlap.append(len(set(ti[i]) & set(pi[i])) / k)
        informative = np.abs(truth[i, ti[i]]) > 1e-8
        direction.append(float((np.sign(pred[i, ti[i][informative]]) == np.sign(truth[i, ti[i][informative]])).mean())
                         if informative.any() else np.nan)
    return pd.DataFrame({"mse_delta": np.mean(error ** 2, axis=1),
                         "mae_delta": np.mean(np.abs(error), axis=1), "pearson_delta": corr,
                         "topk_overlap": overlap, "topk_direction": direction})


def paired_interval(frame, reference, candidate, repeats=2000, seed=17):
    # Average seeds and batch groups before resampling perturbation IDs.
    x = frame.loc[frame.model.isin([reference, candidate])]
    means = x.groupby(["perturbation", "model"]).mse_delta.mean().unstack("model")
    if reference not in means or candidate not in means:
        return None
    means = means.dropna(subset=[reference, candidate])
    if len(means) < 2:
        return None
    difference = (means[candidate] - means[reference]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(difference, len(difference), replace=True).mean()) for _ in range(repeats)]
    return {"reference": reference, "candidate": candidate, "delta_mse": float(difference.mean()),
            "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
            "n_perturbations": len(difference), "n_contexts": int(x.context.nunique()),
            "interpretation": "negative favors candidate; conditional on evaluated contexts, not a cross-context population CI"}


def evaluate_run(run_dir, include_test=False):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    cfg = manifest["config"]
    data = load_prepared(cfg["data_dir"])
    if data["audit"]["fingerprint"] != manifest["data_fingerprint"]:
        raise ValueError("Run/data fingerprint mismatch")
    splits = ["val", "test"] if include_test else ["val"]
    frames, diversity = [], []
    for seed in cfg["seeds"]:
        with np.load(run_dir / f"seed_{seed}" / "predictions.npz", allow_pickle=False) as f:
            if str(f["data_fingerprint"].item()) != manifest["data_fingerprint"]:
                raise ValueError("Prediction fingerprint mismatch")
            for split in splits:
                idx = data["splits"][split]
                for name in f.files:
                    if name == "data_fingerprint":
                        continue
                    metrics = row_metrics(f[name][idx], data["delta"][idx], cfg["top_k"])
                    metrics = pd.concat([data["meta"].iloc[idx].reset_index(drop=True), metrics], axis=1)
                    metrics["seed"], metrics["model"] = seed, name
                    frames.append(metrics)
                groups = sorted({n.split('/')[0] for n in f.files if '/' in n})
                for group in groups:
                    members = [n for n in f.files if n.startswith(group + '/') and n.split('/')[-1] not in {'mean', 'valmix'}]
                    for na, nb in combinations(members, 2):
                        a, b, truth = f[na][idx], f[nb][idx], data["delta"][idx]
                        ea, eb = (a - truth).ravel(), (b - truth).ravel()
                        corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 1e-12 and eb.std() > 1e-12 else np.nan
                        diversity.append({"seed": seed, "split": split, "group": group, "a": na, "b": nb,
                                          "prediction_disagreement_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                                          "error_correlation": corr})
    frame = pd.concat(frames, ignore_index=True)
    columns = ["mse_delta", "mae_delta", "pearson_delta", "topk_overlap", "topk_direction"]
    # Aggregate batch groups, then targets, then contexts. No cell-count weighting.
    units = frame.groupby(["seed", "split", "model", "context", "perturbation"])[columns].mean().reset_index()
    contexts = units.groupby(["seed", "split", "model", "context"])[columns].mean().reset_index()
    summary = contexts.groupby(["seed", "split", "model"])[columns].mean().reset_index()
    overall = summary.groupby(["split", "model"])[columns].agg(["mean", "std"]).reset_index()
    overall.columns = ["_".join(x).rstrip("_") for x in overall.columns]
    overall = overall.sort_values(["split", "mse_delta_mean"])
    out = run_dir / ("evaluation_with_test" if include_test else "evaluation_validation")
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "per_group.csv", index=False)
    contexts.to_csv(out / "per_context.csv", index=False)
    summary.to_csv(out / "per_seed.csv", index=False)
    overall.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(diversity).to_csv(out / "diversity.csv", index=False)
    comparisons = []
    pairs = [("supervised/mean", "kd/mean"), ("kd/mean", "mutual/mean"),
             ("kd/mean", "contrastive/mean"), ("mutual/mean", "mutual_contrastive/mean"),
             ("supervised/valmix", "mutual_contrastive/valmix"), ("teachers/mean", "kd/mean")]
    selection_path = run_dir / "agent/selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection:
        pairs.append((selection["best_fixed_trial"] + "/valmix", "agent_selected"))
    for split in splits:
        for ref, candidate in pairs:
            interval = paired_interval(frame.loc[frame.split == split], ref, candidate,
                                       int(cfg["bootstrap_repeats"]), int(cfg["seed"]))
            if interval:
                comparisons.append({"split": split, **interval})
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    if include_test:
        write_json(out / "test_access.json", {"test_scored": True,
                    "warning": "If these results guide another change, this test set has become development data.",
                    "run_fingerprint": manifest["run_fingerprint"]})
    warning = ("SYNTHETIC SOFTWARE TEST — NOT BIOLOGICAL EVIDENCE" if manifest["synthetic"]
               else "Mean-response development benchmark — NOT official VCC scoring")
    selected_text = (f"Agent 已按验证集固定选择：{selection['selected_trial']}。测试集不能重新选模型。" if selection else "")
    body = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>VCell report</title>
    <style>body{{font-family:system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#243447}}
    table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
    th{{background:#eef2f5}}.warning{{background:#fff2cc;padding:16px}}h2{{margin-top:32px}}</style>
    <h1>VCell · {len(cfg['teachers'])} 教师 × {len(cfg['students'])} 学生</h1><p class="warning">{html.escape(warning)}</p>
    <p>{html.escape(selected_text)}</p>
    <p>主指标：相对匹配对照的平均表达变化 MSE，越低越好。Pearson、top-k 指标只是诊断，非正式差异表达检验。
    验证集用于早停及混合权重选择，因此验证结果有选择偏差。std 是随机种子间标准差；单个种子没有标准差。</p>
    <h2>汇总</h2>{overall.to_html(index=False, float_format=lambda x: f'{x:.5f}')}
    <h2>成对比较</h2>{pd.DataFrame(comparisons).to_html(index=False, float_format=lambda x: f'{x:.5f}')}
    <p>置信区间按扰动基因成组重采样；批次和种子先平均。不能代表未见细胞系总体的不确定性。
    未做多重比较校正；应预先指定主要比较并轮换背景验证。</p>
    <h2>互补性诊断</h2>{pd.DataFrame(diversity).to_html(index=False, float_format=lambda x: f'{x:.5f}')}
    <p>误差相关越低不自动代表越好；必须同时改善预测误差。仅使用训练背景的真实标签更新网络参数。
    教师伪标签不能作为评价真值。</p></html>"""
    (out / "report.html").write_text(body, encoding="utf-8")
    print(overall[["split", "model", "mse_delta_mean"]].to_string(index=False), flush=True)
    return out
