# Research status and terminology

Status recorded on 19 October 2026: the project is temporarily paused.
The owner reports that Round27 did not finish. Its code exists, but no completed
Round27 result is asserted. Run the closeout inventory on the server to identify
the last committed stage and preserve the original files for resumption.

This document summarizes research directions. Detailed protocols remain in the
linked round documents; a design document alone is not evidence of execution.
The server inventory reports the actual directories and their completion markers.

## Experiment history

| Phase | Attempts and purpose | Evidence boundary |
| --- | --- | --- |
| Initial pipeline | Server migration, synthetic GPU smoke and real-data pilot; internal teachers and several numerical students | Established an executable workflow; internal learned teachers were not all pretrained foundation models |
| Pretrained teachers / students | scGPT, scFoundation, State SE and Geneformer; Qwen numerical regression, two distilled students and a supervised VCC-data student | Exploratory comparisons; encoder + fitted head differs from native perturbation prediction; teacher pretraining overlap was not fully verified |
| [Rounds 5–8](ROUND5.md) | UCE fifth teacher, Ridge, teacher reliability, weighted losses, pathways, context separation, traditional-tool reviewers and semantic/graph/module inputs | Tested ways to inject biological priors; supported by the individual round protocols rather than a single universal best configuration |
| [Round9](ROUND9.md) | Clean public background data, background pretraining, module controls and diagnosis | Background representation showed development promise; modules were not uniformly beneficial |
| [Rounds 10–11](ROUND11.md) | Background calibration, context holdouts and few-shot diagnostics | Calibration and background learning were separated from unknown-target prediction |
| [Rounds 12–14](ROUND14.md) | More perturbation data, gene relationships, target representation, low-rank interactions and structural controls | Low-rank/relationship directions were studied; no claim of a universally superior final predictor |
| [Round15](ROUND15.md) | Broad core screening, MIXSCALE inspection, gene ID/alias mapping and matched P4 augmentation | Core screen and later stages had separate execution gates; preparation did not imply every expanded-plan stage ran |
| [Round16 real-model pilot](LIVE_DEEPSEEK_PILOT.md) | DeepSeek called real P4 Ridge/factor/reference tools across contexts | Demonstrated legal model invocation, not evidence that online model selection improved accuracy |
| [Rounds 17–18](ROUND18_SCIENTIST.md) | Nested source OOF, model search, zero expert, routing and bounded self-iteration | Model generation worked; fixed/rule/risk/live routing must be distinguished; development scores were generally near Zero |
| [Round19](ROUND19_DUAL_PREDICTOR.md) | Dual DeepSeek training/review loop with a 100-iteration budget | More iterations mostly converged toward conservative predictions |
| [Rounds 20–21](ROUND21_SIGNATURE_TRANSFER.md) | Replicate/control audits, target reliability, same-versus-wrong-target tests and reference transfer | BXPC3 contained reproducible target information; much Repeat gain came from shared context response; cross-cell gains were not established |
| [Round22](ROUND22_SHINKA_EVOLVE.md) | ShinkaEvolve 0.0.7 + DeepSeek generated complete numerical methods with isolated evaluation | Framework integration succeeded; new candidates did not beat the strongest fixed comparator |
| [Round23](ROUND23_EVOLUTION_FRAMEWORK.md) | Core/Expanded Repeat and Transfer pools, public perturbation sources and persistent evolution | Repeat Expanded achieved about 2.32% lower development-confirmation MSE against that round's comparator; not zero-shot or external validation |
| [Round24](ROUND24_MECHANISM_EVOLUTION.md) | Winner replay/repair, mechanism and data ablations, stronger ensemble comparator, bounded module evolution | The earlier advantage narrowed against a stronger comparator; overall MSE and target-specific response quality were separate objectives |
| [Round25](ROUND25_SELF_CORRECTING_EVOLUTION.md) | Matched Shinka, Repair and Meta arms, automatic repair and scientific strategy interface | Code repair worked; Meta decisions were invalid and no new algorithm surpassed the strongest fixed method |
| [Round26](ROUND26_SCIENTIFIC_CRITIC.md) | Validated Scientific Critic decisions, measured hypothesis changes and Pareto methods | 5/5 accepted decisions, 4 measured revisions; no supported hypothesis improvement; small Repair Repeat gain was not robust |
| [Round27](ROUND27_VIRTUAL_CELL_AGENT.md) | Balanced task quotas, Virtual Cell domain catalog, durable research memory, task management and evidence-linked revisions | Implementation delivered; experiment unfinished; no final Round27 performance claim |

Numbers above are historical development summaries from reviewed artifacts.
The new closeout script does not recompute or independently verify them.
Do not rank rounds by absolute MSE across changing data panels and tasks.

## Latest implemented configuration versus latest reviewed experiment

The latest agent implementation is `configs/round27.json`: ShinkaEvolve 0.0.7,
DeepSeek, three arms (`shinka`, `repair`, `meta`), two tasks, five distinct new
behaviors per task/arm, seed 17 and the ordered 1,809-gene panel. Its budgets and
domain/memory contracts are in the Round27 document. These are configuration
limits and targets, not completed work or elapsed-time promises.

Round26 is the latest completed experiment reviewed in this conversation.
Its historical development-confirmation fixed comparator MSE was approximately
0.00442612 for Repeat and 0.00478958 for Transfer. The Repair Repeat result was
approximately 0.00442300, a tiny improvement with an exploratory interval crossing
zero and worse response-specific diagnostics. The correct conclusion remains
that reliable overall scientific improvement has not been demonstrated.
Round26 did not evaluate a new independent whole-line outer test.

## Data and sample accounting

Historical primary contexts include K562, RPE1, HepG2 and Jurkat. Later IFNG
experiments used A549, BXPC3, HAP1, MCF7 and K562; HT29 was reserved. Auxiliary
public sources included Papalexi THP1 and Adamson K562, with distinct conditions
and perturbation mechanisms. Public cell-atlas backgrounds are background
resources, not extra perturbation labels. Dataset versions, filtering and
fit/calibration/query membership differ by round.

There is no single honest project-wide "training sample count": raw cells,
pseudobulk rows, unique targets, technical replicates and context-target pairs
are different units, and reused sources must not be added multiple times.
Use each prepared dataset's audit and frozen protocol to count these units.
The closeout inventory intentionally does not open raw expression matrices or
sealed test labels to invent a combined count.

## Terminology

| Term | Meaning here |
| --- | --- |
| Delta / response | Mean expression after perturbation minus a matched control mean, under the recorded preprocessing |
| Pseudobulk | An aggregate of cells sharing specified experimental metadata; not an independent biological replicate by default |
| Zero / no-change | Predict an all-zero expression delta |
| Mean transfer | Transfer an average response estimated from permitted training sources |
| Background mean | Shared context response without target-specific identity; an important strong Repeat comparator |
| STRING Ridge | A regularized linear response predictor using recorded gene-network/target features |
| Factor rank16/32 | A low-rank numerical prediction head, not a 16B/32B language model |
| Teacher | A model providing training representations or predictions; its labels are not evaluation truth |
| Numerical Qwen student | A Qwen-based model adapted to predict expression values; not proof of biological reasoning |
| Repeat | Supported technical-replicate prediction using the target's measured Rep1 to predict Rep2 |
| Transfer | Prediction with the query cell line excluded from permitted source labels |
| OOF | Out-of-fold predictions made without fitting on that fold's labels |
| Source search | Legal development evidence exposed to algorithm generation and scientific criticism |
| Confirmation | Frozen retest; historically reused confirmations are still development data |
| Specificity gap | Recorded diagnostic of target discrimination; it must be interpreted with amplitude and controls |
| Pareto retention | Keep methods trading off MSE and response-specific diagnostics rather than a single score |
| Scientific Critic | An LLM making evidence-linked research decisions; a valid decision need not improve prediction |
| Research memory | Provenance-linked source experiments and hypotheses; not new biological knowledge verified by the LLM |
| Framework pass | An engineering/coverage criterion, distinct from prediction superiority |
| Scientific pass | A round-specific metric gate, distinct from independent replication or deployment qualification |

## Closeout decision

Preserve completed methods and unfinished Round27 state. Do not automatically
promote the latest candidate, lowest cross-round MSE, or a partial Round27 method
to a final production model. An exported model is a research artifact with
explicit task, gene panel, dependencies and validation status.
