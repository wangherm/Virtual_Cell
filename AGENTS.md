# Project working rules

- Develop and perform basic validation locally; do not assume local datasets or GPUs.
- AutoDL is a GPU execution backend at `/root/autodl-tmp/virtual_cell`.
- Do not assume remote access. Label instructions LOCAL or AUTODL.
- Keep original data in `data/raw/` immutable. Write derived data to `data/processed/`.
- Keep datasets, checkpoints, logs, outputs and credentials out of Git.
- Maintain both smoke-test and full-training configurations.
- Before full training, pass a small GPU smoke test; inspect shapes, loss and GPU memory.
- Centralize paths, fix random seeds, save the effective config and hyperparameters per run.
- Implement checkpoint resume and keep training separate from evaluation.
- Run long AutoDL jobs inside tmux.
- Inspect existing modules before editing; prefer simple, readable research code.
- For major implementation tasks, report changes, local checks, exact AutoDL smoke-test
  command, expected output, possible failures and the next experiment.
- This repository currently contains scaffolding only; do not describe templates as
  implemented models, data loaders or training pipelines.
