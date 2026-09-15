# Study design and pre-registration

Canonical statement of what this study covers, what it asks, and what would
refute each claim. Adopted 2026-09-15 (DECISIONS.md D-011). The blueprint in
`docs/blueprint_2026-09-09.md` remains the reference for claim discipline,
measurement protocol, and statistical design; where the two differ, this
document governs. Numeric thresholds marked "provisional" are frozen after
the pilots (EXP-002 to EXP-004) and before any confirmatory run; the freeze
is a dated decision entry.

## 1. Scope and the one sentence

Transistor Operations for Machine Learning (TOML) counts the physical events
an algorithm causes (arithmetic by operation type, memory words moved by
level, GPU commands dispatched, fused sequential steps) and maps them to
energy with a small set of device coefficients. This study asks whether one
such model, calibrated per device on a handful of primitives, predicts the
measured training and inference energy of the major machine learning
algorithm families, including reinforcement learning agents, and whether it
predicts the operating regimes where the energy-optimal choice flips in
directions that FLOPs-based reasoning gets wrong.

What the model predicts: energy per unit of executed work (a training step
or epoch, an inference call, an environment step). What it does not predict:
how many units of work an algorithm needs to reach a quality target. Every
hypothesis below is about the former. Energy-to-target results are measured
and decomposed, never claimed as model predictions.

## 2. Platforms and the held-out-platform protocol

- rtx4090-laptop (Windows 11, RTX 4090 Laptop GPU, 175 W cap): development
  and calibration platform. Energy source: power integral (D-009).
- a100-sxm4-40gb (Linux, SLURM): test platform. Energy source: decided by
  its own gate run.

Rule: for every experiment that carries a spine prediction, the predictions
for the A100 (crossover locations, orderings, or intervals) are written into
LOGBOOK.md with a commit hash before the A100 job is submitted. Within a
platform, model selection uses calibration configurations only and the
held-out configurations are scored once by the analysis script; nobody
tunes on them.

## 3. Algorithm catalog

Tier 1 is confirmatory on both platforms; training and inference where both
exist. Tier 2 runs only after Tier 1 is complete and clean. Each family is
included because it exercises a mechanism that FLOPs cannot represent.

| Family | Tier 1 | Tier 2 | Regimes exercised | Implementations |
|---|---|---|---|---|
| Linear and GLM | OLS (normal equations, QR), ridge, Lasso (coordinate descent), logistic regression (SGD, L-BFGS), linear SVM | elastic net, Poisson GLM | dense compute; sequential coordinate updates | torch, scikit-learn, cuML |
| Kernel and instance based | kernel SVM (RBF, SMO), kNN (brute force), Gaussian process regression | kernel ridge, KDE | N-squared kernel and distance matrices, N-cubed Cholesky, irregular CPU search | torch, scikit-learn, cuML |
| Trees and ensembles | CART, random forest, gradient boosting (XGBoost histogram) | LightGBM, extra trees | irregular access, conditional computation, histogram builds | scikit-learn, XGBoost CPU and GPU, cuML RF |
| Clustering and mixtures | k-means (Lloyd, minibatch), GMM (EM), DBSCAN | spectral, agglomerative | iterative convergent; irregular (DBSCAN) | torch, scikit-learn, cuML |
| Decomposition | PCA (randomized SVD), NMF, ICA | t-SNE, dictionary learning | solver overhead, iterative multiplicative updates | torch, cuML |
| Probabilistic and sequential | Gaussian naive Bayes (minimal control), HMM (forward-backward, Viterbi) | CRF | near-zero-compute control; fused sequential | torch, custom |
| Neural, supervised | MLP, CNN (ResNet-18 scale), LSTM and GRU, transformer encoder, decoder-only transformer (GPT-2 small scale), GCN and GraphSAGE | MoE layer, diffusion UNet | dense compute; memory-bound small batch; fused sequential at B=1; attention N-squared; sparse gather and scatter | torch |
| Neural, unsupervised | autoencoder, VAE | contrastive (SimCLR), JEPA | as above | torch |
| Reinforcement learning agents | tabular Q-learning (control), DQN with double and dueling ablations, REINFORCE, A2C, PPO (discrete and continuous), TD3, SAC | model-based (Dyna-style), offline (IQL), CMA-ES | dispatch-bound environment stepping, replay-buffer traffic, small-batch policy inference, batched updates; the number of parallel environments is the batch knob | torch, Gymnasium (classic control, LunarLander, MuJoCo), envpool for vectorized and Atari |

Tier 1 has 34 algorithms; Tier 2 about 14. A raw-feature linear classifier
is a control, not a representation learner, and is not counted as one.

### Primitive experiments (the spine)

Exact-function pairs, one mechanism each, measured before the catalog:

| ID | Pair | Mechanism | FLOPs prediction | Physical prediction |
|---|---|---|---|---|
| EXP-002 | dense versus factorized low-rank GEMM | dispatch and memory terms at small batch | factorize when r < d/2, any batch | crossover rank rises with batch toward d/2 |
| EXP-003 | dense versus factorized, training (backward) | as above with gradient traffic | same rule | separate crossover |
| EXP-004 | dense versus sparse (unstructured CSR, 2:4 structured) | format overhead, indexing | energy proportional to density | dense wins below a density threshold that depends on batch and format |
| EXP-005 | fused versus looped sequential (RNN at B=1, scan) | fused-sequential term | same FLOPs, same energy | fused term dominates at B=1 |
| EXP-006 | dense versus flash attention | memory term | same FLOPs | memory-bound regime decides |
| EXP-007 | same classical algorithm on GPU versus CPU | dispatch versus throughput | GPU wins when FLOPs are large | size crossover N* from separate calibrations |

### Tasks and data

Frozen and hashed splits (data/checksums.sha256): Covertype and a synthetic
generator with controlled N, d, rank, sparsity (tabular and primitives);
CIFAR-10 (images); UCI HAR raw signals (sequences); WikiText-2
(transformers); Cora and ogbn-arxiv (graphs); CartPole, LunarLander,
HalfCheetah, Hopper, Pong (RL).

## 4. Predictors and baselines

| ID | Predictor | Purpose |
|---|---|---|
| P0 | FLOPs (MACs) with one fitted scale | the familiar reference |
| P1 | free-coefficient linear model on ops, bytes, launches, batch, shape | rich conventional baseline; for single-operation workloads it coincides with P4 in structure |
| P2 | gradient-boosted trees and a small MLP on the P1 features | flexible black box |
| P4 | TOML: fixed op-cost table, four fitted device coefficients (compute, memory, dispatch, fused sequential) | the model under test |
| P5 | P4 plus a learned residual | isolates the value of flexible correction |

Matched training data, splits, and tuning budgets. A target run's measured
energy or latency is never an input to its own prediction.

## 5. Research questions and hypotheses

Thresholds are provisional until frozen after the pilots.

### RQ1. Prediction across families

Does one device-calibrated four-term model, calibrated on a small set of
primitives, predict measured training and inference energy across the
catalog, and how does it compare with FLOPs and fitted predictors?

- H1a. Calibrated on at most ten primitive workloads, the model predicts
  held-out algorithms' energy per unit of work with median absolute
  percentage error at most 20 percent and within-family rank correlation at
  least 0.9. Refuted if either bound fails on the held-out families.
- H1b. FLOPs mispredicts dispatch-bound and fused-sequential algorithms by a
  factor of at least 2, with error systematic by regime. Refuted if FLOPs is
  within 2x or its errors are unpatterned.
- H1c. The fixed op-cost ratios of the TO table improve prediction over a
  free-coefficient model of equal size only where many operation types are
  present. Refuted if the free model matches the table everywhere. This is
  the test of the table itself; single-operation experiments cannot test it.

### RQ2. Crossovers (the spine)

Can the model predict, before measurement on a held-out shape, batch, or
platform, where the energy-optimal realization flips, in cases where FLOPs
predicts the wrong side?

- H2a, rank (EXP-002, EXP-003). The crossover rank r*(B) between dense and
  factorized rises with batch size toward the FLOPs value d/2; at B=1 it is
  below d/2 by a factor of at least 2. Refuted if measured r*(B) falls
  outside the model's predicted interval on held-out shapes or FLOPs is
  within 2x at B=1.
- H2b, sparsity (EXP-004). The density threshold below which dense beats
  sparse execution is predicted within the model's interval, including the
  case where unstructured sparsity never wins. Refuted otherwise.
- H2c, batch and dispatch in RL. Energy per environment step is flat in the
  number of parallel environments up to a platform-specific n_env*, below
  which it is independent of policy-network size; the model predicts n_env*
  within a factor of 2 on the held-out platform. Refuted if energy per step
  scales with FLOPs at small n_env or the prediction misses by more than 2x.
- H2d, GPU versus CPU (EXP-007). For iterative classical learners, the
  problem size N* at which the GPU implementation becomes lower-energy than
  the CPU one is predicted within a factor of 2 from separately calibrated
  coefficients. Refuted otherwise. Depends on CPU energy measurement being
  available on each platform (a gate item).
- H2e, sequential (EXP-005). For RNN inference at B=1 and Viterbi, the
  fused-sequential term predicts energy that FLOPs and dispatch counts
  together cannot. Refuted if a three-term model without it fits as well.

### RQ3. Transfer

Do coefficients calibrated on one platform transfer to another with a small
calibration budget?

- H3a. With A100 coefficients fitted on at most five primitives, prediction
  error on the A100 catalog is within 5 percentage points of full A100
  calibration. Refuted otherwise.
- H3b. The ratio of dispatch to fused-sequential coefficients differs across
  platforms in the direction TOMLSignals observed (about 14x on the laptop,
  about 1x on the A100); registered before the A100 runs.
- H3c. Every RQ2 crossover location is predicted for the A100 and filed with
  a commit hash before it is measured there.

### RQ4. Reinforcement learning

Where does RL training energy go (environment stepping, policy inference,
learning updates) across value-based, on-policy, and off-policy methods,
and does the model predict the composition?

- H4a. For n_env at most 16, dispatch overhead is the largest component of
  energy per environment step on both platforms, and the model predicts
  each component's share within 15 percentage points.
- H4b. Off-policy methods with replay (DQN, TD3, SAC) have higher
  learning-update energy per environment step than PPO at equal n_env, in
  proportion to their update-to-data ratio as counted in TOs. Refuted if
  the measured ratio is not ordered by the counted TOs.
- H4c (descriptive). The ranking of algorithms by energy-to-target-return
  differs from the ranking by steps-to-target on at least one of the five
  environments; reported with confidence intervals across seeds and
  decomposed by the predicted per-step energy, not claimed as a prediction.

### RQ5. Controller (gated)

Only if a model-based controller beats a tuned fixed schedule, a contextual
bandit, greedy progress per joule, and model-predictive control under
matched budgets (blueprint H5 and H6, gate G4). Otherwise the paper does not
mention a controller.

## 6. Gates

| Gate | Evidence to continue | If it fails |
|---|---|---|
| G0 instrumentation | EXP-001 passes on the platform | repair, re-run under a new ID |
| G1 mechanism | at least one primitive experiment (EXP-002 to EXP-007) yields a prospectively correct crossover prediction | refine; the catalog alone is a weaker paper and we say so early |
| G2 model value | P4 beats P0 on the held-out catalog and is within reach of P2 | revise the claim |
| G3 transfer | H3a and H3c hold on the A100 | restrict the claim to one platform |
| G4 controller | RQ5 conditions | drop the controller |
| G5 net benefit | quality preserved, overhead accounted | conditional or negative report |
| G6 reproduction | Ameera reproduces a laptop result on her cluster with matched or explained differences | resolve and document |

## 7. Compute plan

Pilots: EXP-002 about 9 hours per platform; EXP-003 to EXP-007 sized after
EXP-002. Catalog: about 1000 windows per platform at 20 s plus warmups,
roughly 16 hours per platform, spread over weeks. Allocation is frozen
before the confirmatory catalog starts.

## 8. Claim discipline

Not claimed: hardware-independent joules from counts alone; transistor
switching from fitted coefficients; a physical lower bound from the best
tested configuration; universal hardware generalization from two NVIDIA
platforms; preserved quality from a point estimate; whole-system savings
from GPU telemetry; net savings without collection, calibration, and
controller accounting; a representation-science contribution from relabeled
hardware counters; award-level significance from catalog size.

## 9. Candidate memorable findings

The data chooses. The design is built to reach: a merge-or-factorize rule
as a function of batch size that contradicts the FLOPs rule; a dispatch
floor below which RL energy per step does not depend on the network; a
GPU-versus-CPU size rule for classical learners. If none survives the
pilots, the catalog and RQ1 stand as the paper, and that decision is taken
after the pilots, not after the catalog.
