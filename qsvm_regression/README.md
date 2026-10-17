# SC4055 QSVM Regression Benchmark

This self-contained experiment predicts vehicle fuel efficiency (MPG) from the
UCI Auto MPG dataset. It compares linear, RBF, and degree-3 polynomial SVR with
ideal and depolarizing-noise quantum kernels using `rot2dof` and `belis` maps.
It includes 5-fold nested cross validation and a separate holdout of the latest
two model years (1981–82). The reference paper's reproducibility approach is
used as a guide, while the dataset and regression task are different:
Kakavand, Strohmeyer, and Schlotter (2026),
[arXiv:2604.18837](https://arxiv.org/abs/2604.18837).

## Run

From this directory, install the project and run the original Auto MPG experiment:

```powershell
python -m pip install -r requirements.txt
python -m qsvm_regression.experiment --smoke --output-dir results/auto_mpg_smoke
python -m qsvm_regression.experiment
```

## Multi-dataset benchmark

The expanded study keeps the Auto MPG experiment and adds three different UCI
regression tasks: Concrete Compressive Strength, Energy Efficiency (Heating
Load), and hourly Bike Sharing total demand (`cnt`). It compares linear, RBF,
and degree-3 polynomial SVR with ideal and depolarizing-noise Rot2DoF and Belis
quantum kernels. Numeric and categorical preprocessing, PCA, and target
standardization are fitted inside each training partition.

From this directory, run a reduced end-to-end check or the full nested-CV run:

```powershell
python scripts/run_benchmark.py --smoke
python scripts/run_benchmark.py
python scripts/analyze_benchmark.py
```

The configurations are `configs/auto_mpg.yaml` and `configs/benchmark.yaml`. Datasets above 500 rows
use a reproducible target-quantile-stratified sample of 500; the same sample is
used for every model. Auto MPG also retains the latest-two-model-years holdout.
Bike Sharing component counts (`casual`, `registered`) and row identifiers are
excluded from the predictors to prevent target leakage. No hardware runs are
included.

The benchmark writes combined and per-dataset fold results, aggregate metrics,
run metadata, paired tests with Holm adjustment, cross-dataset ranks, learning
curves, spectral diagnostics, plots, and `findings_report.md` under
`results/multi_dataset_benchmark/`. Learning curves use fixed parameters
selected by nested CV on the full first outer-fold training partition and are
reported as descriptive diagnostics.

Dataset citations: Auto MPG, https://doi.org/10.24432/C5859H; Concrete
Compressive Strength, https://doi.org/10.24432/C5PK67; Energy Efficiency,
https://doi.org/10.24432/C51307; Bike Sharing,
https://doi.org/10.24432/C5W894. All are fetched from the UCI Machine Learning
Repository at runtime and retain their CC BY 4.0 licensing.

The full run writes fold metrics, aggregate metrics, run metadata, and a plot to
`results/auto_mpg_regression/`. UCI data is fetched at runtime using dataset ID
9. `car_name` is excluded as an identifier; missing predictor values are
imputed within each training fold. The two quantum noise conditions use exact
depolarizing density-matrix simulation (not finite-shot sampling).

To run the focused checks:

```powershell
python -m pip install -e ".[test]"
python -m pytest tests -q
```

Dataset citation: Quinlan, R. (1993). *Auto MPG*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5859H (CC BY 4.0).
