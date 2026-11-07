# When does a quantum kernel help an SVM — and can we predict it before training?

SC4055 group project. Binary classification with quantum-kernel SVMs (simulated
with exact statevectors, shot noise, and Qiskit Aer depolarizing noise) versus
tuned classical SVMs.

**Research question:** under which conditions does a quantum kernel beat a classical
kernel, and do cheap *pre-training* diagnostics (kernel-target alignment, model
complexity / geometric difference, spectrum) predict the winner?

## Motivation

Small clinical datasets are a natural setting for kernel methods: samples are
scarce, features are few, and measurement error is common. Quantum feature maps
promise richer kernels, but reported advantages mostly come from datasets
constructed to favour them or from weakly tuned baselines. We ask a practical
question: when, if ever, is a quantum kernel worth using on a small dataset, and
can we tell before training? We compare matched-budget quantum and classical SVMs
on synthetic and real data, and test whether kernel alignment and geometric
difference predict the winner. We make no claim of clinical value; Heart Disease
serves as a realistic small, noisy tabular case.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt     # installs the package in editable mode
pytest                               # 37 tests: simulator vs Qiskit, kernel properties, leakage, diagnostics, dataset variants
```

Heart Disease is downloaded once from the UCI repository (via `ucimlrepo`) and cached in `data/cache/`.

## Reproduce

```powershell
python scripts/run_experiments.py --experiment all --smoke   # ~1 min pipeline check
python scripts/run_experiments.py --experiment all           # full protocol, results/full/<experiment>/results.csv
python scripts/analyze.py --run full                         # tables, figures, results/full/report.md
python scripts/screen.py --run full                          # screening tool on all 7 datasets -> results/full/tables/tool_table.tex
```

Everything (datasets, grids, folds, noise levels) is in [`configs/default.yaml`](configs/default.yaml).
Each `results/<run>/<experiment>/metadata.json` records the config, git commit, and package versions.

## Design

| Piece | Choice |
|---|---|
| Datasets | three regimes: `moons` (classical-friendly), `ad_hoc` (Havlicek-style, quantum-friendly positive control), `heart` (UCI Cleveland, 303 rows, real-world). Balanced subsample of ≤200 rows. The `suite` experiment adds 30 variants (moons noise × seed, ad hoc circuit × dimension × label noise, heart seed × qubits). |
| Preprocessing | impute → one-hot → standardise → PCA(n = qubits) → min-max to [0, π]; **fit on the training partition only**, including the inner tuning folds (tested in `tests/test_core.py`). `ad_hoc` is generated directly in n dims and used unscaled. |
| Quantum feature maps | `zz` (Havlicek ZZFeatureMap), `he` (RY + CZ ladder), `product` (RY only, separable control). One feature per qubit, tunable `bandwidth` (feature scale), `reps`, `entanglement`. |
| Kernel | fidelity kernel \|⟨ψ(x)\|ψ(y)⟩\|² |
| Classical baselines | linear, RBF (γ tuned), polynomial (γ, degree tuned) |
| Fair tuning | every model: inner 3-fold CV over C ∈ {0.1, 1, 10, 100} × one kernel hyper-parameter (9 values; quantum: bandwidth, RBF: γ) |
| Evaluation | 5×5 repeated stratified CV (nested), accuracy / F1 / AUC, mean ± std over repeats |
| Significance | Nadeau–Bengio corrected resampled t-test (folds share training data), Holm correction |
| Diagnostics (training fold only) | centered kernel-target alignment, `log s_K(λ)` model complexity, geometric difference `g(K_RBF‖K_Q)`, effective rank, off-diagonal variance |
| Predict-then-verify | over the 30 suite variants: does the alignment gap (best quantum − best classical, training data only) predict the accuracy gap? AUC, Spearman ρ with bootstrap CIs, within-family correlations, and a one-parameter switching rule evaluated leave-one-seed-out and leave-one-dataset-out against always-RBF, always-ZZ and an oracle. |

### Experiments

| `--experiment` | Question |
|---|---|
| `main` | Quantum vs classical on the three datasets at 4 qubits (+ diagnostics) |
| `size` | Does the gap change with training-set size? |
| `qubits` | Accuracy and diagnostics for 2–8 qubits |
| `concentration` | Kernel concentration (off-diagonal variance) vs qubits and bandwidth, incl. the `1/√n` bandwidth rule (label-free) |
| `hyperparams` | ZZ bandwidth × reps × entanglement with bandwidth *not* tuned |
| `noise` | Ideal vs finite shots (100/1k/10k) vs Aer depolarizing noise (3 levels) |
| `runtime` | Gram-matrix cost vs qubits and sample size |
| `suite` | 30 dataset variants for the predict-then-verify test and the switching rule |

### Screening tool

`qsvm_clf.screening.recommend_kernel(X, y)` turns the switching rule into something a practitioner can run
before training: it preprocesses the table, picks each kernel's hyper-parameter by training alignment, and
recommends the best quantum map only if its alignment beats tuned RBF by more than `tau = 0.0063`
(the rule's threshold fitted on all 30 suite variants; `python scripts/screen.py --fit-tau` recomputes it).
No SVM is trained.

```python
from qsvm_clf.screening import recommend_kernel
recommend_kernel(X, y, n_qubits=4)   # {'recommendation': 'rbf', 'delta': ..., 'tau': 0.0063, ...}
```

## Scope and honest limitations

* Kernels are simulated; "quantum advantage" claims are about *kernel quality on small data*, not runtime.
* The Aer noise kernel is the Hilbert–Schmidt overlap Tr(ρ(x)ρ(y)) of noisy states, a hardware-agnostic proxy for a swap-test; it is not a full compute–uncompute circuit and omits readout error and connectivity.
* One realisation of each dataset is drawn (seed 0); repetition is over CV splits, so stds reflect split variance, not dataset variance.
* ≤ 200 samples and ≤ 8 qubits (simulation cost, and kernels concentrate beyond that without bandwidth control).
* The predict-then-verify test uses a modest number of units; its correlations are suggestive, not definitive.
