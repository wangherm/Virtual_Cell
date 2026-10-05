# Round15 core-screen review

Reviewed 2026-10-05 from the user-supplied `round15_review_light.tar.gz`.
This is a review of development results, not a new experiment or external test.

## Integrity and scope

- 2,385 archive members, including its archive manifest. All 2,384 payload hashes
  matched. Reaggregation of the per-target table reproduced the mean comparison
  table to a maximum absolute discrepancy of 9.8e-17.
- 225 newly completed heads, 63 reused results and 36 blocked slots. The reuse
  counts and blocked definitions match the predeclared available core screen.
- The evaluation uses three optimizer seeds on the same previously inspected
  development splits, not three independent biological replications. P3/P4/P5/P6
  and few-shot stages were not run. No new Mixscale data entered these models.
- The public IFNG RDS passed its size/MD5 checks, but D1 inspection was blocked
  because the dedicated R environment failed to install from overseas channels.
- This light archive omits weights and numerical prediction arrays. The review
  checks tables and included evidence; it does not independently recompute MSE
  from model predictions or rerun the biological training.

## Main comparisons

Lower MSE is better. Compare methods within a column; the three tasks have different
response distributions. Values below are equal-context macro averages.

| Method | New context / seen target | Seen context / new target | H1 double holdout |
|---|---:|---:|---:|
| Zero change | 0.02839229 | 0.01858697 | **0.00495881** |
| Historical ID Ridge | **0.02594700** | 0.01788747 | 0.00760438 |
| Historical STRING Ridge | 0.02799216 | 0.01772756 | 0.00966127 |
| Local factor, no ID, rank32 | 0.02713059 | 0.01762856 | 0.00613637 |
| Local factor, no ID, rank16 | 0.027091 | **0.017591** | 0.006065 |
| Control PCA factor | 0.027697 | 0.017626 | 0.006124 |
| Repaired ID-only + background | 0.026682 | 0.017884 | 0.005442 |
| Learning rate 1e-4 | 0.027760 | 0.017942 | 0.005148 |
| Ridge residual | 0.035752 | 0.018512 | 0.061830 |
| Neighbor-response residual | 0.029604 | 0.019314 | 0.010416 |

### Correct local gene relations have a clearer development signal

The no-ID rank32 factor beats all three matched shuffled-relation controls for
unknown targets. Absolute paired MSE gains are approximately 0.000242, 0.000207,
and 0.000284, with the three exploratory 95% intervals above zero. These intervals
are uncorrected for the broad screen and are not independent confirmation.
TSR2 remains an influential target, but removing the largest contributor leaves
positive gains of approximately 0.000155, 0.000116, and 0.000193.

Rank16 improves mean MSE over rank32 by **0.213%**, winning on 45/66 unique targets
after averaging seeds and available contexts. The paired absolute-gain interval
is approximately [0.000011, 0.000068]. Against historical STRING Ridge its mean
gain is 0.00013661 (~0.77%), but an independently recomputed 8,000-draw paired
target bootstrap (seed 7815) gives [-0.0000418, 0.0003427], which crosses zero.
Rank16 is a confirmation candidate, not a proven replacement for Ridge.

MSE improvement did not improve every specificity metric: rank16 macro Top-1 is
7.88%, versus rank32 9.06%; specificity gap is 0.0370 versus 0.0399. These are
macro averages of context-specific retrieval panels, not one common candidate set.

### Background representation is not settled by this screen

Control PCA and frozen B2 have almost identical unknown-target mean MSE:
0.017626 versus 0.017629. Their paired gain interval crosses zero. B2 is better
in the new-context column, so this does not establish PCA superiority. It does
show that the current unknown-target score alone cannot justify a claim that
the learned background encoder is necessary. Keep PCA as a cheap matched baseline.

### The repaired ID control now learns, chiefly where targets are seen

For new context / seen targets, repaired ID-only + background improves over
background-only from 0.027747 to 0.026682 (~3.84%). The exploratory paired interval
is positive. It remains worse than historical ID Ridge. Its small changes for
unknown targets can arise from jointly changed training of the background branch;
the unknown ID embedding is still zero and these scores do not prove unseen-ID
information is available.

### H1 double holdout remains a failure

Learning rate 1e-4 reduces the rank32 error substantially but still exceeds zero
change by **3.82%**, winning on only 6/140 targets against zero. Its macro Top-1
is 0.238%, below the 140-candidate random expectation of 0.714%. Mean prediction
RMS falls from 0.03415 to 0.01510, consistent with a more conservative response.
This is not evidence of recovered perturbation specificity or correct biology.
All nine low-learning-rate heads selected the final 2,000-step checkpoint, so
the comparison also reflects optimization progress at a fixed update budget.

Ridge and neighbor-response residuals have poor cross-context MSE in their current
implementations. Ridge residual reaches 0.06183 in double holdout, around 12.47
times the zero baseline, with a large prediction amplitude. It uses a fixed-alpha
OOF baseline, not necessarily the same fitted Ridge as the historical comparator.
The failure does not rule out every residual method, but this implementation
should not be promoted. Neighbor retrieval can improve specificity in K562 while
worsening MSE, so its ranking score alone is not sufficient for adoption.

The new cosine, contrastive, gene-scale weighting and weight-decay variants do
not provide a material, consistent improvement over the MSE parent. Contrastive
training had eligible rows; its weak result is not simply an inactive loss.

## Next decisions

1. Complete D1 inspection and validate cell line, guide/modality, stimulus,
   study/replicate and matched controls. Keep all HT29 sources/stimuli reserved.
2. Confirm rank16 versus rank32, matched no-ID shuffled relations, simple control
   PCA, and historical Ridge on new development partitions with fold-specific
   fitted transforms. Freeze a small set of comparisons before running them.
3. Prioritize additional cross-context perturbation coverage for the double task.
   Do not spend the next budget multiplying current residual or loss variants.
4. Retain simple baselines. There is no single model supported as best on all tasks.

The recorded ~4.09-hour orchestration wall time includes data transfer and setup;
it is not GPU training time. The 225 new heads have median recorded training time
~7.8 seconds and maximum per-process PyTorch allocation ~92 MiB. These measurements
describe the small heads, not full Qwen/native-model training or total device memory.
CPU/RAM values in `hardware.json` may reflect the host, not container quotas.
