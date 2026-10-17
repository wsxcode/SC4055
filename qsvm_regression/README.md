# QSVM Regression Benchmark

This project compares classical SVR kernels with ideal and depolarizing-noise
quantum kernels on four UCI regression datasets: Auto MPG, Concrete Compressive
Strength, Energy Efficiency (Heating Load), and hourly Bike Sharing demand
(`cnt`). All four are run through the same benchmark pipeline. Auto MPG is a
dataset in the combined study; there is no separate Auto MPG experiment.

## Run

From this directory:

```powershell
python -m pip install -r requirements.txt
python scripts/run_benchmark.py --smoke
python scripts/run_benchmark.py
python scripts/analyze_benchmark.py
```

The dataset list and experiment settings live in `configs/benchmark.yaml`.
Datasets with more than 500 rows use a fixed-seed, target-stratified sample,
shared by every model. Auto MPG also has a latest-two-model-years holdout.
Imputation, encoding, scaling, and PCA are fit inside each training fold. Bike
Sharing component counts and row identifiers are excluded to prevent target
leakage. UCI datasets are fetched at runtime; no raw dataset files are bundled.

The combined results are written to `results/multi_dataset_benchmark/`.
Per-dataset fold files are grouped in `datasets/` with one file per dataset.
Combined tables are in `tables/`, plots in `figures/`, run provenance in
`metadata/`, and the written summary in `report/`.

The tables include combined fold and aggregate metrics, paired statistics,
cross-dataset ranks, and learning-curve measurements. The metadata records
dataset provenance, sampled row indices, configuration, software versions,
and seeds.

Dataset sources: [Auto MPG](https://doi.org/10.24432/C5859H), [Concrete
Compressive Strength](https://doi.org/10.24432/C5PK67), [Energy
Efficiency](https://doi.org/10.24432/C51307), and [Bike
Sharing](https://doi.org/10.24432/C5W894). The evaluation design is guided by
Kakavand, Strohmeyer, and Schlotter (2026),
[arXiv:2604.18837](https://arxiv.org/abs/2604.18837).
