# Characterizing and Controlling the Energy Cost of Learned Representations

## TOML-based ICLR study blueprint

Prepared for Muntaser Syed · 9 September 2026

**Research status:** proposed definitions, hypotheses, experiments, and decision criteria. No new experimental results are claimed in this document. Best-paper ambition sets the standard for the investigation; it is not a prediction of an award or acceptance.

**Available equipment:** an RTX 4090 laptop with approximately 16 GB GPU memory, access to one A100 40 GB, and possibly a second A100. The laptop's cumulative NVML energy counter was previously tested. Current access, isolation, driver versions, and controllable settings on the A100 still need verification.

**Scope:** classical and neural ML; training and inference; representation characterization, physical validation, transfer, and reinforcement-learning control. LLMs and transformers are optional additional cases. They do not define the study.

## 1. The scientific contribution to pursue

The main question is:

> Which properties of learned representations explain the physical cost of learning and using them, under what execution conditions do these characterizations transfer, and can a controller exploit that knowledge to improve measured energy efficiency at preserved task quality?

The strongest prospective contribution is a compact, operational characterization that predicts new cases, survives controlled interventions, identifies its own limits, and supports useful decisions. Increasing the number of algorithms or adding an RL controller does not establish that contribution by itself.

Distinguish **realized cost** from **realizable efficiency**. Low effective rank or activation sparsity may have little direct effect on a dense kernel. Those properties may instead predict which lower-cost realization or learned transformation retains task utility. The central characterization should therefore predict attainable energy-quality trade-offs over a declared implementation/transformation set, as well as realized energy where possible.

Organize the paper around a causal chain:

1. Define the representation and its computational realization.
2. Characterize representation properties and execution costs separately.
3. Predict energy and the conditions under which rankings change.
4. Test these predictions with interventions at controlled quality.
5. Use the characterization to guide representation learning and execution.
6. Demonstrate transfer and net energy savings after all optimization costs.

The intended claim is conditional: a useful representation-level characterization exists over a declared set of algorithms, implementations, devices, and workload distributions. A universal scalar assigning joules to an abstract representation is not assumed.

## 2. What must be new relative to prior work

Energy-aware optimization, hardware-aware compression, and learned energy prediction already exist. For example, [energy-aware pruning](https://arxiv.org/abs/1611.05128) uses hardware-grounded energy estimates to guide compression; [NeuralPower](https://arxiv.org/abs/1710.05420) predicts CNN runtime and energy; [HAQ](https://arxiv.org/abs/1811.08886) uses RL and hardware feedback for quantization; and [Zeus](https://arxiv.org/abs/2208.06102) optimizes training configurations. The paper must distinguish its contribution from these lines explicitly.

[Representation similarity research](https://proceedings.mlr.press/v97/kornblith19a.html) provides tools for comparing learned features, but representational similarity is not itself an electrical-energy measurement. [Sponge Examples](https://arxiv.org/abs/2006.03463) also establishes that inputs can affect energy or latency in studied systems. Neither input-dependent energy nor failure of FLOPs as a universal energy proxy should be presented as a new discovery here.

TOML's existing [Signals implementation](https://github.com/jemsbhai/TOMLSignals) separates compute, memory, command-dispatch, and fused-sequential terms. Reuse and cite these components transparently. The new paper must identify the additional representation characterization, validated transfer behavior, and control method. Existing TOML data can support development or a clearly labeled retrospective analysis; fresh, untouched experiments are needed for confirmatory claims.

The literature checks supporting this blueprint are a targeted scoping review, not an exhaustive novelty certification. Before freezing the contribution, maintain a comparison matrix covering energy modeling, hardware-aware compression, energy-aware AutoML, sparse/low-rank representation learning, constrained RL, and hardware transfer. Compare actual assumptions and evaluation protocols, not titles alone.

## 3. Define the objects precisely

Use distinct notation for two different meanings of representation:

- **Task representation:** z = f_theta(x), such as a learned projection, codebook assignment, sparse code, tree-leaf encoding, or neural feature vector.
- **Execution descriptor:** phi(f_theta, k, D), a TOML-based description of the work performed by an implementation k on workload distribution D.

The RL controller may learn its own internal state representation. That is a third object and must not be confused with the task representation being studied.

For a fixed task representation and readout g, specify the operational context

\[
c=(h,k,D,b,p,u),
\]

where h is the hardware platform; k includes library, compiler, kernel, layout, and fusion choices; D is the input distribution; b is batching/concurrency; p is precision; and u contains the operating policy and relevant measured state, including clocks and temperature.

Define expected inference energy per completed example as

\[
E_{\mathrm{use}}(f,g\mid c)
=\mathbb E_{x\sim D}[E_{\mathrm{physical}}(k\text{ executing }g(f(x)))]
\]

with an explicitly declared measurement boundary. For batched execution, measure batch energy and divide by the number of completed examples; do not imply that every example's energy is directly observed.

Report the encoding and readout contributions where they can be measured without materially changing execution, and always retain an end-to-end measurement. Materializing a latent tensor for analysis may break fusion or add memory traffic; diagnostic and deployed execution paths must be distinguished.

Define training energy to a target quality as the total energy from the declared start to the first scheduled evaluation meeting the target, including validation cost. Impose a maximum resource horizon, retain failed runs, and report success rate. Reaching an easy target is not evidence of retaining the quality of a stronger baseline.

Average power is energy divided by elapsed time. Power efficiency should be reported as useful throughput per watt, equivalently useful work per joule under a consistent boundary, with quality and latency specified. Low average watts alone do not demonstrate efficiency. Report power constraints and their averaging windows separately from energy objectives.

For a deployment with N uses, define a scenario-dependent lifecycle cost:

\[
E_{\mathrm{life}}(N)=E_{\mathrm{learn}}+E_{\mathrm{search/calibration}}+E_{\mathrm{setup}}+N E_{\mathrm{use}}+E_{\mathrm{maintenance}}(N).
\]

Count one-time feature materialization, persistent caching, storage I/O, and repeated retraining in the appropriate scenario. State the assumed N rather than declaring one representation globally optimal.

## 4. Preregister the principal hypotheses and failure conditions

| ID | Hypothesis | Decisive test | What would undermine it |
| --- | --- | --- | --- |
| H1 | Representation properties identify lower-energy feasible realizations at preserved task quality. | Predict held-out energy-quality frontiers and feasible-action rankings, with and without representation features; separately test direct energy prediction at fixed execution conditions. | Descriptors add no value for either realized cost or the attainable quality-constrained frontier against rich baselines. |
| H2 | TOML's structure improves data efficiency and transfer. | Matched predictor capacity, data, features, and calibration budgets; held-out families and platforms. | An unstructured predictor or unweighted primitive counts match the gains with the same budget. |
| H3 | Proposed mechanisms predict the effect of controlled interventions. | Function-equivalent or explicitly quality-matched sparse, low-rank, precision, and basis experiments. | Correlations fail under intervention or are explained by changed quality or workload. |
| H4 | Conditional characterizations predict crossover regimes. | Predict which representation wins on a held-out device, batch regime, or implementation before measuring outcomes. | Crossovers are explained only retrospectively or predictions are uncalibrated. |
| H5 | Sequential RL exploits delayed consequences usefully. | Same information and action access for RL, a bandit, greedy control, and model-predictive control. | Simpler methods match performance and adaptation cost. |
| H6 | Optimization provides net benefit. | Full accounting of collection, controller training, calibration, switching, monitoring, and use. | Savings disappear after overhead or require an implausible deployment horizon. |

Freeze primary endpoints, quality tolerances, workloads, excluded configurations, and analysis procedures after the instrumentation pilot but before confirmatory evaluation. Do not choose a desired savings percentage and optimize the analysis until it appears.

H1 is especially important for this paper's framing. A better hardware cost predictor without additional insight into learned task representations does not by itself establish a representation-science contribution.

The two causal pathways to test are (a) representation properties changing executed work or data-dependent hardware behavior, and (b) representation properties determining which cheaper transformations preserve utility. A null direct effect at fixed dense execution does not refute pathway (b). A candidate transformation that truncates a vector only after computing the full expensive encoder must be charged for that encoder; a cheaper encoding path must actually be implemented and validated.

## 5. The theoretical agenda

### 5.1 Establish the boundary of the characterization

A learned function or its geometric similarity class does not uniquely identify an executable program. Equivalent functions may use different factorizations, storage layouts, or algorithms. Therefore define energy relative to a permitted implementation class and operating context.

Formalize this boundary using ordinary, useful implementations. An example that only adds deliberate redundant work proves little of scientific interest. This definitional observation is a supporting result, not the main novelty claim.

For a prescribed quality threshold q, one possible operational frontier is

\[
\mathcal F_h(q)=\mathrm{Pareto}\{(E,T,M): (f,g,k)\in\mathcal C,\ Q(f,g)\ge q\},
\]

where the implementation class C, workload, and measurement boundary are stated. An empirically measured frontier is the frontier of tested feasible configurations; do not call it a global physical lower bound.

### 5.2 Separate computational structure from device calibration

Investigate a model of the form

\[
\widehat E_h=\beta_h^{\mathsf T}\phi(f,k,D,b,p)+r_\eta(f,k,D,b,p,u,h).
\]

The descriptor can include precision-specific operations, data movement, command overhead, supported sparsity structure, and additional validated TOML terms. The residual captures effects not represented accurately by the structured model. Nonnegative total energy must be enforced.

Treat fitted coefficients as effective cost parameters. Regression coefficients do not directly identify capacitances or individual transistor switching events. Correlated predictors may make individual coefficients nonidentifiable even when prediction is good. Use controlled calibration workloads and report parameter uncertainty and conditioning.

TOML remains a relative structural framework. The hardware-conditioned model is a separate bridge to measured joules. Changing a power cap can change energy without changing the operation count, so counts alone cannot be the complete control model.

### 5.3 Derive a crossover prediction that can be tested

For an exactly rank-r linear map W of shape d_out by d_in, compare computing Wx with computing U(Vx), where W=UV. The two produce the same mathematical output. The approximate arithmetic counts per example are d_in*d_out versus r*(d_in+d_out), but the factorized realization may add intermediate memory traffic, commands, or inefficient matrix shapes.

Under a declared additive primitive-cost model, factorization is preferable when its arithmetic savings outweigh its additional memory and execution overhead. Derive the crossover condition as a function of batch size, rank, shape, precision, and calibrated costs. Test predicted crossover locations on held-out settings. Verify floating-point output differences against a predefined numerical tolerance.

The algebraic cost comparison is elementary and related ideas are established. A contribution would require accurate prospective crossover prediction, applicability beyond one operator, and demonstrable decision value across representation families.

### 5.4 Develop useful uncertainty statements

For two feasible configurations A and B with descriptors phi_A and phi_B, a conditional dominance certificate can use

\[
\sup_{\beta\in\mathcal B_h}\beta^{\mathsf T}(\phi_A-\phi_B)+\epsilon_A+\epsilon_B<0,
\]

where B_h and the residual bounds are justified on the declared domain. This implies the ordering only on the event that the calibration set and error bounds cover the truth. Quantify empirical coverage and abstention; do not silently extend calibration guarantees under arbitrary distribution shift.

An optional control analysis can relate energy-prediction and transition-model error to decision regret under explicit bounded-horizon and feasibility assumptions. A standard regret lemma is not sufficient novelty. Prioritize a theorem that explains an observed transfer behavior, calibration requirement, or failure condition.

## 6. Representation families and controls

| Family | Concrete representations | Why it matters | Initial status |
| --- | --- | --- | --- |
| Learned linear subspaces | PCA/Incremental PCA; suitable low-rank learned projections | Rank, dimensionality, data reuse, and factorized execution | Core |
| Codebooks and sparse coding | MiniBatch K-means assignment/distance features; dictionary learning or NMF | Sparsity structure, distance calculations, codebook size, iterative learning | Core, one inexpensive member first |
| Tree representations | Gradient-boosted-tree or random-forest leaf/path encodings | Conditional computation, irregular access, encoding and storage costs | Core |
| Neural features | MLP/autoencoder representations, compact CNN, selected RNN | Learned activation statistics and dense/sparse execution | Core |
| Fixed-feature learners | Logistic regression and linear SVM on prescribed features | Negative/control cases without a learned intermediate encoder | Core controls |
| Additional families | GMM responsibilities, kernel approximations, small transformer | Tests beyond the core boundary if justified by results and compute budget | Extension |

A raw-feature linear classifier is not automatically a representation learner. Preserve that distinction. Its learned decision scores may be studied, but including it should not inflate the count of independently validated representation-learning families.

For trees, distinguish native prediction from explicitly materialized leaf embeddings plus a probe. Measure encoding and sparse/dense materialization costs. A dense one-hot expansion can artificially disadvantage a native tree implementation.

Run two complementary quality evaluations:

1. Native model evaluation, preserving each family's normal prediction path.
2. Frozen representation evaluation with a declared, consistently tuned probe protocol.

Account for probe fitting and inference. Use the same data access and tuning budget. Report matched-quality frontiers and equal-dimension slices separately. Equal dimension is not equal utility; a common probe is not a substitute for native performance.

## 7. Public-data design

Begin with three verified public sources and controlled synthetic mechanisms:

| Source | Use | Split requirements |
| --- | --- | --- |
| [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) | Tabular classification; tree, linear, codebook, and neural comparisons | Freeze row IDs and preprocessing. Treat an environmental/wilderness-area shift as a separate labeled stress test, with class-support checks. |
| [CIFAR-10/CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) | Image representation and probe studies | Preserve official test data. Use a validation split from training. CIFAR-10 and CIFAR-100 are related benchmarks, not two independent modality demonstrations. |
| [UCI HAR](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones) | Sensor representations, classical and neural models | Use subject-disjoint evaluation. Treat raw-signal and supplied-feature experiments separately and account for feature extraction. |
| Synthetic rank/sparsity/task generators | Mechanism identification with controlled latent structure | Publish generation code, all seeds, transformations, and noise levels. Never present synthetic-only validation as broad real-world transfer. |

The initial confirmatory matrix can use four named datasets from these sources plus the synthetic families. After the pilot, add independently sourced tabular/regression or temporal tasks if required to test a concrete generalization claim. Freeze those additions before confirmation; do not expand only around favorable results.

Use a compatibility matrix rather than forcing every algorithm onto every task. Within each comparison, inputs, preprocessing access, labels, quality target, and measurement boundary must be matched. If a family is omitted from a dataset, record the reason before viewing confirmatory energy results.

## 8. Representation measurements

Measure a deliberately small initial feature set:

- Dimension and physical storage size.
- Exact and threshold-defined sparsity, with the threshold recorded.
- Block/structured sparsity and supported executable layout.
- Singular-value spectrum and a declared effective-rank definition on a fixed sample.
- Value range, quantization distortion, and precision sensitivity.
- A task-quality or probe-quality measure with confidence intervals.
- Family-specific descriptors: occupancy of codebook entries, leaf/path frequencies, or the cost of evaluating sparse codes.

Use representation similarity, including CKA where appropriate, as a diagnostic. It is not a surrogate energy label. Discrete code entropy needs a defined alphabet; continuous-feature entropy requires an estimator and scale convention. Do not treat arbitrary entropy estimates as directly comparable physical quantities.

Calculate expensive descriptors on frozen subsamples. Record their energy and runtime. Test a cheaper feature subset for deployment. Offline descriptors that require an entire finished training trajectory must not become inputs to an online controller before they are available.

## 9. Controlled experiments that discriminate between explanations

### E1. Identical sparse values, different executable realizations

Use the same mathematical tensor/map and compare dense storage and kernels against a supported sparse realization. Include dense zero masking as a control. Sweep batch size, shape, and sparsity structure prospectively.

The intended test is whether the characterization predicts when sparsity becomes physically useful, including conversion, indexing, and format overhead. Zeros in a dense tensor do not establish that a device skips arithmetic. A null benefit for unstructured sparsity on a particular backend is informative and must remain in the results.

### E2. Exact low-rank factorization and crossover prediction

Construct exact rank-r maps and compare a precomputed dense map with the factorized map. Preserve inputs and mathematical outputs. Verify numerical tolerances. Sweep rank, batch, matrix shape, precision, and device; include intermediate allocations and commands.

Fit the predictor on a calibration subset, then predict held-out crossover regions. Report prediction error in the crossover location and the energy penalty of choosing the wrong realization. An observed reduction in FLOPs alone is not the endpoint.

### E3. Function-preserving changes of basis

For suitable linear latent interfaces, transform z to Qz with orthogonal Q and compensate a linear readout W by WQ^T. Predictions are mathematically unchanged, as are pairwise Euclidean distances of latent vectors. Coordinate sparsity and executable structure can change.

Fold transformations into existing linear maps where mathematically valid. If folding is not valid, include the transformation's full cost and analyze it separately. Do not add an extra operation and then attribute its obvious cost to representation geometry. Do not assume the rotated case changes GPU energy on a fixed dense kernel: whether a measurable effect exists is part of the test.

This experiment examines which geometric properties omit information needed for physical characterization. It does not prove a new universal invariance theorem merely by exhibiting alternative implementations.

### E4. Representation learning interventions at controlled quality

Train representation variants with prespecified rank, codebook, sparsity, or regularization interventions. Use matched tuning budgets and independent seeds. Compare both the native prediction path and a frozen-probe protocol where appropriate.

These interventions generally change the learned function. Use noninferiority/equivalence criteria and matched-quality frontiers; do not label them function-preserving. Record energy to learn the representation and energy to use it. This is the key bridge from execution characterization to choices made during learning.

### E5. Checkpoint trajectories

At fixed progress checkpoints, characterize the learned representation and evaluate its quality and inference energy. Charge diagnostic evaluation to the experimental accounting; distinguish the instrumented research path from a deployable low-overhead controller.

Ask whether representation evolution predicts future useful improvement per joule or final inference cost beyond current loss, elapsed compute, and execution-only features. Prevent future-checkpoint leakage. Split all checkpoints from a training run together.

### E6. Hardware and workload crossover

Take frozen representations and run them on both GPU platforms with matched software where possible. Vary realistic batch/load regimes. Use a supported CPU path when reliable CPU measurement is available.

Predeclare test grids. Predict ordering before measuring test configurations. If the software stack necessarily differs between the Windows laptop and the A100 server, label the result deployment-platform transfer and isolate hardware-only transfer in the subset where software can be matched.

### E7. Negative controls

Include shuffled energy labels for predictor sanity checks, irrelevant/noise descriptors, fixed representations, and actions that alter overhead without altering learning dynamics. These are analysis checks, not additional claims of real-world benefit.

Retain cases in which TOML fails, sparse execution loses, a dense kernel is best, a bandit matches RL, or optimization never recovers its setup cost. The failure conditions define the characterization's useful scope.

## 10. Predictor baselines and ablations

Use the same training/evaluation splits and comparable tuning budgets. Separate pre-execution prediction from prediction using online telemetry or prior profiling.

| ID | Baseline or model | Purpose |
| --- | --- | --- |
| P0 | FLOPs/MACs, parameter count, and latent dimension with fitted scaling | Establish familiar simple references. |
| P1 | Operations plus estimated bytes, launches, batch, precision, shape, layout, and implementation metadata | Avoid a weak FLOPs-only comparison. |
| P2 | GBDT and modest MLP on the full permitted conventional feature set | Test flexible unstructured prediction. |
| P3 | Existing TOML accounting, frozen before the new test data | Measure what prior work already explains. |
| P4 | Calibrated TOML components with no representation descriptors or learned residual | Isolate structure and calibration. |
| P5 | P4 plus learned residual, without task-representation descriptors | Isolate flexible correction. |
| P6 | Representation descriptors plus conventional execution features, without TOML weighting/structure | Test the incremental value of TOML. |
| P7 | Proposed combined model | Test the full hypothesis. |

Match model capacity in comparisons where capacity could explain improvement. Evaluate ablations of rank, element sparsity, structured sparsity, precision sensitivity, and quality features. Report feature-acquisition costs. A target run's measured energy is never an input used to predict that same run's energy. A target run's measured latency cannot be silently treated as a free pre-execution feature.

Primary predictor endpoints:

- Pairwise action-ranking accuracy with measurement uncertainty handled explicitly.
- Decision regret: measured energy selected by the predictor minus the best measured feasible candidate on a declared finite grid.
- Error and coverage under held-out family, dataset, and platform conditions.
- Calibration energy/observations required for a specified decision quality.

Include absolute error and suitable relative error summaries, but avoid unstable percentage metrics near zero. Global correlation across workloads with very different scales is insufficient: include within-workload rankings and stratified errors.

For attainable-frontier prediction, pair the energy model with a quality-change/feasibility model. Test whether representation descriptors predict which transformations retain useful information beyond task identity, current score, dimensions, and conventional metadata. Report false feasibility decisions and missed feasible improvements. Selection on validation data must be followed by untouched quality and energy evaluation; retrospective feasible-oracle results are reference curves only.

## 11. RL as an intervention on representation learning

Keep RL central as the adaptive mechanism, with representation characterization as the scientific basis. The controller should be able to influence meaningful learning decisions as well as execution settings. Otherwise a successful controller might establish only a runtime scheduling result.

Develop a small constrained model-based controller first. Compare its policy learning with alternatives using the same learned models and observations. A structured energy predictor alone does not provide training-state transitions; learn and validate progress/transition models separately.

Use an observation history when loss and representation summaries are not a sufficient state. Do not declare the process fully Markov merely because those summaries are convenient.

### 11.1 Two groups of actions

**Representation-learning actions** can include valid update-allocation decisions, codebook-learning effort, sparsity-regularization schedules, or supported precision schedules. Rank or dimension changes should enter only when state migration, optimizer state, and implementation behavior are defined and measured.

**Execution actions** can include batching, supported device power settings, and choices among validated kernels/layouts. Construction, conversion, compilation, and switching costs must be included.

Build algorithm-specific adapters that expose feasible actions and their semantics. Do not invent dynamic control support in a library. A parameter accepted at initial construction may not be safely changeable within an existing run. For boosting, for example, changing histogram construction or future-tree behavior requires an explicitly supported and verified protocol.

Start with two controllable families having genuine delayed consequences, such as an MLP representation learner and MiniBatch K-means, then add boosting or an incremental decomposition learner after the adapter is validated. Characterization can cover more families than the initial online-control experiment; report the distinction.

### 11.2 Factor the control experiment

Evaluate four conditions under matched budgets:

1. Tuned fixed representation-learning schedule and execution settings.
2. Adaptive execution settings with representation-learning choices fixed.
3. Adaptive representation-learning choices with execution settings fixed.
4. Both action groups available.

This identifies the source of improvement. Use the same quality objective and action availability for competing controllers.

### 11.3 Training objective

Minimize total energy to a prespecified useful quality target, subject to a completion horizon and other declared constraints. Failed runs remain visible. Consider the lifecycle objective only for explicitly declared deployment horizons N; do not change N after seeing which method wins.

Use measured energy as the final evaluation signal. If TOML supplies model rollouts, shaping, or exploration priorities, anchor them to held-out physical measurements. A policy that reduces predicted TO cost without reducing measured energy does not establish energy efficiency.

Do not use discounted rewards in a way that makes postponing energy consumption appear beneficial. Use an appropriate finite-horizon objective, a clear terminal condition, and penalties/constraints that prevent gaming quality or uncompleted work.

### 11.4 Inference objective

Evaluate a frozen learned representation on a fixed submitted workload, or on a specified arrival process with full queue accounting. Preserve quality, latency, and completion constraints. Include queue drainage or charge unfinished work so that deferral cannot appear as an energy saving.

If the inference action is a single fixed configuration with no important delayed effects, use it as a static-selection/bandit test. Do not manufacture an RL environment by repeatedly presenting independent configuration choices.

### 11.5 Controller baselines

- A well-tuned fixed policy, beyond library defaults.
- Random and Bayesian configuration search for static decisions.
- A contextual bandit for adaptive decisions with no planning horizon.
- A greedy progress-per-joule controller with matched information.
- Model-predictive control using the same learned cost and transition models.
- A standard model-free RL policy under matched interaction budgets.
- A standard model-based RL policy without TOML structure.
- Relevant specialist methods, including Zeus when its recurring-training setting and actions match.
- A retrospective best candidate over a measured finite grid, labeled as a reference and charged appropriately if treated as a deployable search method.

Report calibration/interactions in both physical energy and number of trials. Comparable trial counts alone can hide very different collection costs. Give baseline tuning a fair budget; include those costs when comparing lifecycle performance.

## 12. Transfer protocol and leakage prevention

Use separate axes of generalization:

1. New seeds/checkpoints within a family.
2. New shapes, scales, batches, or quality targets.
3. Entirely held-out datasets.
4. Entirely held-out algorithm families.
5. A held-out deployment platform with a prespecified calibration allowance.

Do not randomly split rows of a telemetry log when neighboring rows belong to the same model/run. Group splits by run, representation artifact, and dataset as needed. All variants derived from one parent representation must stay together unless the explicit experiment is within-parent intervention prediction.

Distinguish zero-shot evaluation, calibration-only adaptation, and policy fine-tuning. State which target-platform information each condition receives. Test labels and final quality outcomes cannot guide adaptation.

An A100 and a laptop 4090 support two-platform evidence. A second A100 supports replication and additional capacity; it does not create a third architecture. If both available platforms influence model selection, neither remains an untouched hardware test. Use nested or independently replicated protocols and describe the limited hardware sample honestly.

The key comparison is a shared controller/descriptor with limited adaptation against family-specific or platform-specific learning from scratch using the same target budget. Merely fitting separate controllers to every family does not demonstrate transfer.

## 13. Measurement protocol

1. Record exact device model and form factor, memory, driver/runtime/library versions, operating system, power/clock settings, precision policy, compilation mode, and other processes.
2. Verify cumulative energy-counter availability, update behavior, supported control settings, and permissions. Do not infer control access from the GPU model name.
3. Prefer isolated full-device measurements for core experiments. Device-level energy counters do not automatically provide per-process or per-tenant attribution.
4. Establish thermal behavior and steady-state operating conditions. Randomize or block run order to avoid coupling one method to cooler hardware.
5. Validate a minimum measurement-window duration using repeated fixed workloads. Polling more frequently does not improve the physical sensor's update resolution.
6. Synchronize at measurement boundaries without imposing artificial synchronization inside the actual algorithm. Account for asynchronous work correctly.
7. Separate setup/compilation from steady-state operation, then include both in end-to-end and lifecycle scenarios. Show amortization explicitly.
8. Measure preprocessing, host-device transfers, encoding, readout, and postprocessing under clearly declared boundaries. A GPU-only energy reduction cannot establish total-system energy reduction.
9. Use reliable CPU energy domains when available and record exactly which domains are covered. Do not substitute TDP times runtime for measured CPU energy in the primary comparison.
10. Use gross device energy as a clearly labeled main operational measure and report idle-subtracted dynamic energy as a sensitivity analysis where useful. Log the idle measurement method and its uncertainty.
11. Treat two APIs reading the same hardware sensor as consistency checks, not independent validation. External metering, if available, validates a different boundary and needs aligned windows.
12. Quantify monitoring and descriptor overhead. For short classical workloads, this overhead may dominate; those cases must not be silently removed.

[NVIDIA's NVML documentation](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) defines the cumulative GPU energy API. [Zeus measurement guidance](https://ml.energy/zeus/measure/) explains measurement windows and synchronization. Neither source guarantees the accuracy of this study's particular setup; validate it experimentally.

## 14. Statistical design

- Use discovery runs to characterize variance and choose meaningful measurement windows. Freeze the confirmatory analysis afterward.
- Start discovery with three independent learning seeds where relevant. Plan at least five independent seeds for primary stochastic results, increasing replication when pilot variance or the desired quality-equivalence margin requires it. Five is a starting allocation, not a universal sufficiency guarantee.
- Distinguish independent training seeds, independent controller seeds, and repeated measurements of one frozen artifact. Thousands of power samples are not thousands of independent experiments.
- Use paired comparisons on common tasks, inputs, and compatible initializations. Randomize method order within measurement blocks.
- Prespecify task-specific quality noninferiority margins and report confidence bounds, not only point estimates. For exact-function tests, prespecify numerical tolerances instead.
- Report effect sizes and uncertainty at the task/run level. Use a hierarchical or cluster-resampling analysis appropriate to the experiment's nesting.
- Report individual tasks and worst/failure cases alongside aggregate summaries. Energy ratios can be summarized on a log scale only for meaningful positive denominators and clearly comparable quality conditions.
- Retain failed quality targets, OOMs, invalid actions, and budget exhaustion. Present success rate and budget-censored cost; do not report only the mean of successful runs as the main outcome.
- Separate confirmatory hypotheses from exploratory correlations. Handle multiple comparisons for formal inferential claims.
- Do not draw one causal conclusion from an observational correlation between representation rank and energy. Use the interventions to identify which mechanism is supported.

## 15. Full cost accounting and break-even analysis

Maintain separate ledgers for:

1. Dataset preprocessing and baseline model/representation learning.
2. Feature extraction and physical profiling for the cost model.
3. Quality-label acquisition for transformation or frontier prediction.
4. Transition-data collection and controller training.
5. New-family or new-platform calibration/adaptation.
6. Runtime monitoring, policy evaluation, and configuration switching.
7. Model use, recurring retraining, and maintenance.

Avoid double-counting shared experiments, but do not omit them. Report both the full research energy bill and a reproducible deployment accounting scenario. An already published calibration set is not physically free; state whether its collection cost is sunk, shared, or amortized in each scenario.

For comparable repeated jobs and positive per-job savings, define

\[
N_{\mathrm{BE}}=\frac{E_{\mathrm{collection}}+E_{\mathrm{controller\ fit}}+E_{\mathrm{initial\ adaptation}}}{E_{\mathrm{baseline/job}}-E_{\mathrm{controlled/job}}}.
\]

Controlled-job energy includes runtime controller and switching overhead. Add recurring adaptation to the per-job denominator or the explicit maintenance function. A nonpositive denominator means no break-even under that scenario. Express uncertainty in break-even estimates, especially when the denominator is small.

For a representation that costs more to learn but less to use, separately report the training-versus-inference crossover. Keep workload lifetime and representation reuse assumptions visible. Raw accuracy divided by joules is not an adequate replacement for matched-quality comparisons.

## 16. Decision gates

| Gate | Evidence needed to continue | Consequence if the evidence fails |
| --- | --- | --- |
| G0: instrumentation | Stable, interpretable measurements; supported controls; accounting boundaries verified | Repair the harness before collecting a benchmark. |
| G1: mechanisms | At least one substantive mechanism or accurately predicted boundary beyond an already-known qualitative effect | Refine the hypothesis; retain null cases. Do not scale a weak correlation study. |
| G2: representation value | Useful improvement in realized-cost prediction or attainable-frontier prediction beyond rich baselines | Revise the representation-level claim; inspect whether the result is only conventional execution modeling. |
| G3: transfer | Prospective performance on untouched families/workloads and bounded platform adaptation | Restrict the claim or improve the method before widening it. |
| G4: sequential value | RL improves an important metric over bandit/greedy/MPC controls under comparable budgets | Keep the characterization result, but stop presenting RL as the established source of benefit. |
| G5: physical and net benefit | Quality preserved, total measured savings demonstrated, overhead and failures reported | Report a conditional or negative result; do not claim operational efficiency. |
| G6: independent reproduction | A collaborator runs the released protocol on a clean setup with matched or explained results | Resolve discrepancies and document environment sensitivity. |

Do not define these gates as an arbitrary required percentage of savings. A small, robust result explaining a broad phenomenon can be more informative than a large gain over an untuned baseline. Best-paper competitiveness depends on what the study discovers, the novelty relative to the full literature, and the strength of the evidence.

## 17. The six main figures to design before experiments

These are figure specifications, not placeholders for invented results.

1. **Representation, implementation, and energy:** paired exact-function or quality-controlled cases showing what the characterization predicts, including a null control.
2. **Energy-quality frontiers:** measured feasible points across classical and neural representation families, with learning cost and use cost clearly separated.
3. **Prospective crossover maps:** predicted versus measured ranking changes over batch/shape/rank/device regimes, with uncertainty and failure regions.
4. **Incremental representation information:** full ablation against conventional execution features and TOML without representation descriptors; include acquisition cost.
5. **Held-out-family and platform adaptation:** performance versus measured calibration energy, comparing shared adaptation and learning from scratch.
6. **Adaptive control and net benefit:** fixed/bandit/MPC/RL comparisons, the representation/execution action factorial, constraint violations, and break-even curves.

One main table should enumerate workloads, algorithms, versions, quality criteria, implementation boundaries, and hardware. A second can summarize each main claim with its direct supporting experiment and limitation.

## 18. Resource strategy for the available hardware

The study does not require training a large foundation model. Use compact models and repeated controlled experiments to obtain statistical and mechanistic evidence.

- Use the laptop for instrumentation, compact models, representation diagnostics, and repeatable measurement sessions.
- Use the A100 for controlled sweeps and the second deployment platform. Request isolated slots and verified control permissions through the collaborator when the experiment requires them.
- Use an optional second A100 for independent repetitions and turnaround. It is not necessary for the core hypothesis.
- Use CPU paths where relevant and measurable. Do not force all classical workloads onto the GPU.
- Reuse a frozen representation across inference platforms when possible. An inference comparison does not require retraining the encoder separately on each GPU.
- Use a learned environment only after checking its counterfactual predictions against real transitions. Static measurement-table replay is not a validated simulator of training dynamics.
- Stop optional scaling experiments when they add no new evidence about a declared claim.

Do not promise a GPU-hour estimate before the pilot. Estimate it from actual fit, profiling, policy-training, and replication times:

\[
H_{\mathrm{study}}=\sum_i n_i t_i/3600+H_{\mathrm{controller}}+H_{\mathrm{calibration}}+H_{\mathrm{reproduction}},
\]

where each t_i is a pilot-measured seconds-per-run estimate and n_i is the planned number of comparable runs. Include failure/restart overhead. Freeze a compute allocation across discovery, confirmatory tests, transfer, and reproduction before the full sweep. Avoid an uncontrolled Cartesian product of every algorithm, task, action, and device.

## 19. Reproducibility package

Release a versioned research package containing:

- A run manifest with configuration, environment, source revision, seeds, parent representation, dataset/split identifiers, and measurement boundary.
- Raw timestamped energy, time, temperature, control settings, and completion records.
- Representation artifacts or reproducible construction instructions, with checksums and licensing information.
- Explicit algorithm-adapter contracts, valid action spaces, and state-migration behavior.
- Separate labels for measured physical quantities, modeled quantities, and offline diagnostics.
- Training/validation/test and cost-model/controller split manifests.
- Predefined analysis scripts and figure-generation recipes.
- Accounting ledgers, failed runs, exclusion reasons, and corrections.
- An independent-reproduction report recording deviations and unresolved discrepancies.

Keep a dated decision log. Record changes to hypotheses and protocols before collecting the affected confirmatory data. Preserve earlier analyses as superseded versions. This makes the paper assessable even when some initial ideas fail.

## 20. Immediate pilot

The next implementation should answer three questions before attempting the complete suite.

**Pilot A: can we predict a real physical crossover?** Construct a set of exact low-rank maps with paired dense/factorized implementations. Measure both GPUs over a prespecified range of shapes and batches. Fit on part of the grid and predict the remaining part. Include conversion and launch overhead and verify numerical equivalence.

**Pilot B: do learned representation properties predict useful cheaper realizations?** On Covertype and one sensor or image task, fit a learned subspace, a codebook representation, a tree representation, and a compact neural representation where the interfaces are practical. Evaluate native outputs and the declared probe protocol. Compare observed feasible frontiers with predictions from conventional features, TOML-only features, and the combined representation-aware model. Keep all diagnostic cases in the report.

**Pilot C: does sequential adaptation help?** For two validated iterative learners, compare a tuned fixed schedule, a bandit/greedy controller, and a small model-based RL controller over a restricted action set. Preserve final quality and include collection and controller energy. Add MPC before claiming planning-specific gains.

At the pilot review, decide which mechanistic prediction is credible, whether representation information adds value, whether two-platform transfer is measurable, and whether RL has earned a role beyond simpler control. Then freeze the confirmatory protocol and broaden the suite.

## 21. Paper framing and claim discipline

Working title: **Characterizing and Controlling the Energy Cost of Learned Representations**.

A possible contribution statement, to be used only if supported by results:

> We characterize the attainable energy-quality trade-offs of classical and neural representations under declared computational realizations, test the mechanisms with controlled interventions, and show that the resulting descriptors support limited-calibration transfer and effective sequential control.

The introduction should explain why characterizing a representation's computational possibilities matters for learning decisions. The formalism should state its implementation and measurement boundary. The experiments should establish the mechanism, its generalization, and its practical use in that order.

The paper should leave readers with one memorable, falsifiable finding. Candidates include a transferable condition predicting when compact representations save energy, an accurately predicted reversal across execution contexts, or a small representation descriptor that improves quality-preserving adaptation to unseen families. These are candidate discoveries, not predetermined conclusions.

Do not claim:

- Hardware-independent absolute joules from representation geometry alone.
- Exact transistor switching measurements from software-level counts or fitted coefficients.
- An intrinsic physical lower bound from the best tested configuration.
- Universal hardware generalization from two NVIDIA platforms.
- Preserved quality from a point estimate alone.
- Whole-system savings from GPU telemetry alone.
- Net savings without collection, calibration, and controller accounting.
- A learning-representation contribution from relabeling conventional hardware counters.
- Award-level significance based on benchmark size or a collection of standard components.

The ambition is to establish a reusable scientific characterization and show its consequences. The final title, claims, and emphasis must follow the observed evidence.

## 22. Primary reference map

These links support the prior-art and implementation discussion; they do not establish the proposed results.

| Source | Role in the study |
| --- | --- |
| [TOML Signals code and experiment artifacts](https://github.com/jemsbhai/TOMLSignals) | Existing structured cost model and measurement foundation; clearly separate reused work from new contributions. |
| [TOML/NEXUS framework](https://github.com/jemsbhai/nexus-ml) | Existing framework context and interfaces to audit before extension. |
| [Kornblith et al., representation similarity](https://proceedings.mlr.press/v97/kornblith19a.html) | Representation-comparison methodology and its distinction from execution cost. |
| [Yang et al., energy-aware pruning](https://arxiv.org/abs/1611.05128) | Prior energy-aware transformation and compression. |
| [NeuralPower](https://arxiv.org/abs/1710.05420) | Prior learned energy/runtime prediction. |
| [HAQ](https://arxiv.org/abs/1811.08886) | Prior RL-based hardware-aware quantization. |
| [Zeus](https://arxiv.org/abs/2208.06102) | Prior training energy optimization and relevant baseline. |
| [Perseus](https://arxiv.org/abs/2312.06902) | Analytical optimization of distributed-training energy; scope-match before direct comparison. |
| [Energy-Aware Dynamic Neural Inference](https://arxiv.org/abs/2411.02471) | Prior adaptive inference under energy constraints. |
| [Sponge Examples](https://arxiv.org/abs/2006.03463) | Prior input-dependent energy/latency effects; do not assume the same effect sizes or mechanisms on the proposed hardware. |
| [Scikit-learn incremental learning](https://scikit-learn.org/stable/computing/scaling_strategies.html) | Algorithm interfaces and important differences in batching semantics. |
| [XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html) | Supported GPU execution and implementation considerations. |
| [cuML estimator documentation](https://docs.nvidia.com/cuml/latest/estimator_intro/) | Available classical ML GPU implementations. |
| [NVIDIA NVML queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) | Counter semantics and availability checks. |
| [NVIDIA NVML controls](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html) | Supported control settings and privilege requirements. |
| [Zeus measurement documentation](https://ml.energy/zeus/measure/) | Measurement-window and synchronization practices. |
| [ICLR subject areas](https://iclr.cc/Conferences/2026/CallForPapers) | Venue scope; no award or acceptance criterion is inferred from this page. |

Before claiming novelty, inspect the full text and artifacts of the closest prior studies and refresh the literature matrix. A recent [RL power-control preprint](https://arxiv.org/abs/2608.11226) appeared in the preceding scoping search, but only its indexed abstract was accessible; treat it as an unresolved related-work item rather than a fully assessed comparison.
