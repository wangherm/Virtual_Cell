# Round 13: diffusion, low-rank interactions and a seen-target residual

Stage A freezes the Round12 expanded data and B2 background encoders. It tests
small models before any new Qwen tuning. No additional perturbation labels,
atlas cells, teachers, calibration mixture or confidence gate are introduced.

## Why this experiment

Round12 expanded STRING Ridge improved target-holdout MSE from 0.018026 to
0.017728 and beat the three shuffled mappings in exploratory target bootstrap
comparisons. Double-holdout MSE was 0.009661 versus zero's 0.004959, with no
target-level wins over zero among 140 H1 targets. These are development results.

Two interpretations need care:

- Round12 STRING Ridge **already used a full target/background outer product**.
  Round13 tests representation coverage, low-rank factorization and memory
  regularization; it does not claim to introduce interaction for the first time.
- Removing a target's ID branch does not withhold its response labels from the
  STRING branch. Target masking is an ID-dependence regularizer, not a substitute
  for the existing whole-target outer holdout.

## Start and monitor

The complete original Round12 directory and its dependencies must still exist,
including prepared data, baseline prediction arrays and background encoders.
The light archive alone cannot run this stage. The default parent is
`/root/autodl-tmp/vcell-work/runs/round12_01`.

```bash
cd /root/autodl-tmp/virtual_cell && \
git -c http.version=HTTP/1.1 pull --ff-only && \
screen -dmS vcell13 bash scripts/run_round13_autodl.sh --name round13_01 --resume
```

```bash
/root/autodl-tmp/virtual_cell/.venv/bin/python \
  /root/autodl-tmp/virtual_cell/scripts/round13_status.py

tail -n 40 -f "$(ls -t /root/autodl-tmp/vcell-work/logs/round13_pipeline_*.log | head -n 1)"
```

The initial stages verify Round12 checksums, download approximately 85.1 MB
(81.2 MiB) of pinned STRING files and compute graph diffusion. They are CPU/disk
work; low GPU activity is expected. Existing Qwen model configuration is local,
and no Hugging Face download or backbone training is needed.

Defaults: three seeds (17/29/43), three original protocols, nine workers with
two workers active at once and two CPU math threads each. There are 14 new heads
per worker, 126 total; 63 are small neural heads, the remainder are Ridge fits.
All nine background encoders and original baseline predictions are reused.
The neural update budget is 2,000 steps, batch 128, hidden width 32, AdamW with
learning rate 0.0005 and weight decay 0.01. Checkpoints are evaluated/saved every
100 steps using source calibration labels only. Outer data never selects a
checkpoint, representation, rank or post-hoc scale.

Optional `--prepare-only` completes graph preparation and writes jobs. Resume
without that flag to train. `--parallel-jobs 1` reduces concurrency. A changed
model budget, code or input requires a new run name. Five GiB free space is the
preflight floor; nothing is deleted automatically.

Downloads resume verified byte ranges and must match committed size/SHA256.
Background and graph artifacts are checksum-verified. Each finished head is
reused on resume; unfinished Ridge heads restart, and neural heads recover the
optimizer, best calibration checkpoint and exact update/mask sequence. A
completed graph is reused. Interrupted diffusion recomputes from the small
pinned download cache; it never triggers a new raw-expression download.

## Model comparisons

| Arm | Representation | Interaction / memory |
|---|---|---|
| Historical baselines | Zero, mean transfer, Round12 no-target/ID/STRING Ridge | Exact cached predictions |
| panel_additive_ridge | Round12 panel STRING | Additive only; contrast with historical outer-product Ridge |
| diffusion_additive_ridge | Full-network diffusion | Additive only |
| diffusion_interaction_ridge | Full-network diffusion | Full outer product, same Ridge alpha grid |
| undiffused_interaction_ridge | Graph's initial random vectors | Controls the retained random/self component |
| diffusion_additive | Full-network diffusion | Small bounded additive neural model |
| diffusion_factor | Full-network diffusion | Bounded low-rank target × background interaction |
| diffusion_factor_id | Full-network diffusion | Add fit-target ID residual |
| diffusion_factor_id_dropout | Full-network diffusion | Add whole-target ID masking at probability 0.5 |
| shuffled_diffusion_1/2/3_interaction_ridge | Three shuffled mappings | Same Ridge pipeline |
| shuffled_diffusion_1/2/3_factor_id_dropout | Three shuffled mappings | Same final neural model and training budget |

The neural model uses:

```
h_p = tanh(U_p * target + mask(target) * ID_residual(target))
h_c = tanh(U_c * background + b_c)
coefficient = W_p * h_p + W_c * h_c + W_pc * (h_p * h_c) + b
delta = fit_only_response_basis.T * coefficient * fit_only_response_scale
```

The additive neural control omits the cross term but uses the same bounded
activations, initialization and optimization. Consequently stability improvements
are not automatically attributed to the interaction term. Output layers start
at zero response. ID residuals start at zero, only fit targets have trainable
IDs, and index zero is the disabled unseen-target path. One ID mask is sampled
for the full fit-target vocabulary per optimizer step, shared across every row
and background for that target. Masks are disabled at inference for seen IDs;
unseen IDs remain disabled permanently.

Ridge selects alpha from the same source-calibration grid as Round12. Neural
models select the minimum calibration coefficient MSE checkpoint among saved
steps. With the fixed orthonormal basis, minimizing this coefficient error is
equivalent to minimizing full-delta MSE up to a constant and scale. No new
post-hoc shrinkage or outer-label early stopping is used. The historical MLP
used a fixed endpoint; do not attribute every difference from it to architecture.

## Graph definition and provenance

The [official STRING v12 human download page](https://version-12-0.string-db.org/cgi/download?species_text=Homo+sapiens)
provides the exact files pinned in `configs/string_v12_human.json`. Data are
credited to the STRING Consortium under CC BY 4.0; the modifications here are
confidence filtering, symmetric graph construction, diffusion and projection.
The release stays at v12.0 to avoid conflating a database update with the method.

All human protein nodes are retained. Edges use combined confidence >=700/1000;
reciprocal AB/BA records are not added twice. This is a high-confidence graph
over the full human network, not every low-confidence STRING edge and not a
signed causal regulatory graph.

Let T be its weighted row-normalized adjacency and R be a fixed 64-dimensional
random projection, seed 1301. Iterate H = 0.15 R + 0.85 T H until the contraction
error bound is <=1e-6, with at most 150 iterations. H is a random projection of
personalized PageRank rows, not the full dense protein-by-protein PPR matrix.
Its self component is retained and has an undiffused random control. Isolated
nodes are explicitly missing evidence, represented by zero plus a coverage flag.

Targets map through the existing canonical-symbol table to exact STRING
preferred names. Multiple connected proteins for a symbol are averaged uniformly;
no fuzzy alias or annotation guessing occurs. Per-fold projection to 32 target
dimensions fits only fit-target vectors. Graph construction itself uses public
topology, including proteins corresponding to held-out targets, but never their
perturbation-response labels. This is transductive use of public biology.

The local real-file check found 19,699 proteins, 236,930 undirected retained
edges, convergence in 94 iterations, and 672/704 targets mapped to connected
nodes. Server `mapping.csv` is authoritative for its pinned run. Missing targets
remain in every all-target metric. Shuffling mappings controls gene/profile
alignment; it does not by itself isolate a causal effect of topology. Public
network evidence may overlap published perturbation biology.

## Outputs and interpretation

Download only:

`/root/autodl-tmp/vcell-work/runs/round13_01/round13_review_light.tar.gz`

Main reports:

- `comparison_mean.csv`, `comparison.csv`: mean and per-seed scores.
- `context_scores.csv`: MSE, specificity gap, tie-aware Top-1/Top-5, median rank
  separately by context. Raw ranks across different candidate counts are not
  directly comparable; normalized ranks are also provided.
- `paired_comparisons.csv`: matched target bootstrap differences, win counts,
  and removal of the largest-gain target.
- `oracle_sweep.csv`: fit-only rank-32/rank-64 projection diagnostics. Neither is
  a usable prediction; rank64 is not selected from the outer result automatically.
- `background_shift.csv`: fit/calibration/outer latent norm diagnostics.
- `graph_evidence/mapping.csv`, `audit.json`: actual graph coverage and provenance.
- `splits/*/membership.csv`: exact inherited fit/calibration/outer rows.
- `report.html`, `evaluation_scope.json`: overview and interpretation limits.

All scores preserve the original equal-context/equal-target/equal-row weighting.
Bootstrap resamples target identities jointly across contexts after averaging
optimizer seeds. Intervals are exploratory and uncorrected for the many tested
models. Optimizer seeds are not independent biological samples.

Beating zero on double holdout while increasing own-versus-other correlation is
a useful development milestone, not proof of solving unseen-gene/new-background
generalization. Require paired robustness, shuffled controls and new independent
contexts before making that claim. No automatic winning model or Qwen stage B
is launched. The next data priority, if needed, is overlapping perturbations
across independent backgrounds rather than merely adding unique single-line targets.
