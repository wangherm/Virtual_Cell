"""One observe -> propose -> train -> measure loop; no agent framework."""
from pathlib import Path
from typing import Literal
import json
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from .train import load_run, run_trial, source_fingerprint, settings_for, save_predictions
from .utils import write_json

# Fixed, visible search space. Epoch budget and architecture stay constant across trials.
class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    learning_rate: float = Field(ge=1e-5, le=3e-3)
    weight_decay: float = Field(ge=0, le=.01)
    kd_weight: float = Field(ge=0, le=1)
    peer_weight: float = Field(ge=0, le=.5)
    contrast_weight: float = Field(ge=0, le=.2)
    teacher_weights: list[float] = Field(min_length=1)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["train", "stop"]
    assessment: str
    rationale: str
    settings: Settings | None


SYSTEM = """You manage a small perturbation mean-response training experiment.
Observe validation summaries and training histories; propose ONE controlled next trial or stop.
Minimize valmix_mse averaged across seeds, while inspecting individual MSE and ensemble gain.
Mean-transfer and no-change are real baselines. Greater diversity alone is not success.
History supervised/kd/peer/contrastive are normalized losses; val_normalized_mse is normalized
equal-mean ensemble validation error. Final validation.json scores use original delta units.
Teachers are frozen reference networks, not pretrained foundation models.
Use the same architecture, seeds, splits, and epoch cap as prior trials; each trial restarts.
Allowed strategies: adjust learning_rate/weight_decay; adjust kd/peer/contrast weights;
change nonnegative teacher_weights in listed order (sum exactly 1; zero can exclude a teacher).
Prefer changing few parameters to make effects interpretable. Avoid strong claims from small
validation gains or synthetic data. Validation is adaptively reused and may overfit.
Explain assessment and rationale in Chinese. For stop use settings=null.
You have no tools for source edits, shell commands, datasets, or test results."""


class LLMPlanner:
    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI(max_retries=0)
        self.model = model
        self.last_call = None

    def __call__(self, observation):
        response = self.client.responses.parse(
            model=self.model, store=False, max_output_tokens=3000,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": json.dumps(observation, ensure_ascii=False, allow_nan=False)}],
            text_format=Decision)
        self.last_call = {"response_id": response.id, "model": response.model,
                          "usage": response.usage.model_dump() if response.usage else None}
        if response.output_parsed is None:
            raise RuntimeError("Agent returned no complete decision; inspect API refusal/incomplete response")
        return response.output_parsed


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_decision(decision, n_teachers):
    if decision.action == "stop":
        if decision.settings is not None:
            raise ValueError("Stop requires settings=null")
        return None
    if decision.settings is None:
        raise ValueError("Train requires settings")
    weights = np.asarray(decision.settings.teacher_weights)
    if (len(weights) != n_teachers or not np.isfinite(weights).all() or
            (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-6, rtol=0)):
        raise ValueError("teacher_weights must match teachers, be nonnegative and sum to 1")
    settings = decision.settings.model_dump()
    settings["teacher_weights"] = (weights / weights.sum()).tolist()
    return settings


def trial_summary(run_dir, cfg, name, settings):
    records = []
    for seed in cfg["seeds"]:
        path = Path(run_dir) / f"seed_{seed}" / name
        summary = read_json(path / "validation.json")
        history = pd.read_csv(path / "history.csv")
        cols = ["epoch", "supervised", "kd", "peer", "contrastive", "val_normalized_mse"]
        sampled = sorted(set([0, int(history.val_normalized_mse.argmin()),
                              *range(max(0, len(history) - 5), len(history))]))
        records.append({"seed": seed, **summary, "history": history.iloc[sampled][cols].to_dict("records")})
    scores = [r["valmix_mse"] for r in records]
    return {"name": name, "settings": settings, "score": float(np.mean(scores)),
            "seed_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None, "seeds": records}


def observation(run_dir, cfg, state, max_trials, synthetic):
    trials = [trial_summary(run_dir, cfg, mode, settings_for(cfg, mode)) for mode in cfg["experiments"]]
    trials += [trial_summary(run_dir, cfg, t["name"], t["settings"]) for t in state["trials"]]
    return {"synthetic": synthetic, "objective": "mean_over_seeds_of_validation_macro_MSE_valmix",
            "teacher_order": [s["name"] for s in cfg["teachers"]], "students": cfg["students"],
            "teacher_validation": [read_json(Path(run_dir) / f"seed_{s}" / "teacher_validation.json") for s in cfg["seeds"]],
            "baselines": [read_json(Path(run_dir) / f"seed_{s}" / "baseline_validation.json") for s in cfg["seeds"]],
            "student_epoch_cap": cfg["student_epochs"], "patience": cfg["patience"],
            "warmup_epochs": cfg["warmup_epochs"], "temperature": cfg["temperature"],
            "remaining_trials": max_trials - len(state["trials"]),
            "prior_decisions": [t["decision"] for t in state["trials"]], "trials": trials}


def run_agent(run_dir, model, max_trials=3, resume=False, planner=None):
    """planner injection exists for tests; CLI always uses the LLM."""
    run_dir = Path(run_dir)
    manifest, data = load_run(run_dir)
    if source_fingerprint() != manifest["source_fingerprint"]:
        raise ValueError("Source changed since baseline run; rerun with current code")
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    cfg = manifest["config"]
    torch.set_num_threads(int(cfg["num_threads"]))
    out = run_dir / "agent"
    state_path = out / "state.json"
    identity = {"run_fingerprint": manifest["run_fingerprint"], "model": model, "max_trials": max_trials}
    if state_path.exists():
        if not resume:
            raise FileExistsError("Agent run exists; use --resume")
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("Agent model or budget changed; start a new experiment")
        if state["finished"]:
            return read_json(out / "selection.json")
    else:
        state = {"identity": identity, "trials": [], "finished": False, "pending": None}
        write_json(state_path, state)
    # Initialize API client only when an actual new decision is needed.
    while len(state["trials"]) < max_trials:
        if state["pending"] is None:
            obs = observation(run_dir, cfg, state, max_trials, manifest["synthetic"])
            number = len(state["trials"]) + 1
            write_json(out / f"observation_{number:03d}.json", obs)
            if planner is None:
                planner = LLMPlanner(model)
            decision = planner(obs)
            settings = validate_decision(decision, len(cfg["teachers"]))
            record = {"decision": decision.model_dump(), "api": getattr(planner, "last_call", None)}
            write_json(out / f"decision_{number:03d}.json", record)
            if decision.action == "stop":
                state["stop_reason"] = decision.rationale
                break
            state["pending"] = {"name": f"agent_{number:03d}", "settings": settings, **record}
            write_json(state_path, state)
        pending = state["pending"]
        run_trial(run_dir, pending["name"], pending["settings"], data, cfg, resume=True)
        state["trials"].append(pending)
        state["pending"] = None
        write_json(state_path, state)
    obs = observation(run_dir, cfg, state, max_trials, manifest["synthetic"])
    best = min(obs["trials"], key=lambda t: t["score"])
    best_fixed = min(obs["trials"][:len(cfg["experiments"])], key=lambda t: t["score"])
    selection = {"selected_trial": best["name"], "validation_mse": best["score"],
                 "best_fixed_trial": best_fixed["name"], "best_fixed_validation_mse": best_fixed["score"],
                 "validation_gain": best_fixed["score"] - best["score"],
                 "trials_completed": len(state["trials"]),
                 "stop_reason": state.get("stop_reason", "trial budget exhausted"),
                 "student_model_epoch_cap": max_trials * len(cfg["students"]) * len(cfg["seeds"]) * cfg["student_epochs"],
                 "checkpoints": [str(run_dir / f"seed_{s}" / best["name"] / "best.pt") for s in cfg["seeds"]],
                 "interpretation": "adaptive validation selection; held-out evaluation is still required"}
    for seed in cfg["seeds"]:
        folder = run_dir / f"seed_{seed}"
        with np.load(folder / "predictions.npz", allow_pickle=False) as f:
            selected = f[f"{best['name']}/valmix"]
        save_predictions(folder, {"agent_selected": selected}, data)
    write_json(out / "selection.json", selection)
    write_json(out / "final_observation.json", obs)
    state["finished"] = True
    write_json(state_path, state)
    from .evaluate import evaluate_run
    evaluate_run(run_dir)
    return selection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optional training agent")
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_agent(args.run, args.model, args.max_trials, args.resume), ensure_ascii=False, indent=2))
