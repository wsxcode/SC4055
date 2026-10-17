# Auto MPG Quantum Kernel Regression

This self-contained experiment predicts vehicle fuel efficiency (MPG) from the
UCI Auto MPG dataset. It compares linear, RBF, and degree-3 polynomial SVR with
ideal and depolarizing-noise quantum kernels using `rot2dof` and `belis` maps.
It includes 5-fold nested cross validation and a separate holdout of the latest
two model years (1981–82). The reference paper's reproducibility approach is
used as a guide, while the dataset and regression task are different:
Kakavand, Strohmeyer, and Schlotter (2026),
[arXiv:2604.18837](https://arxiv.org/abs/2604.18837).

## Run

From this directory, install the focused dependencies and run the experiment:

```powershell
python -m pip install -r requirements.txt
python experiment.py --smoke --output-dir results/auto_mpg_smoke
python experiment.py
```

The full run writes fold metrics, aggregate metrics, run metadata, and a plot to
`results/auto_mpg_regression/`. UCI data is fetched at runtime using dataset ID
9. `car_name` is excluded as an identifier; missing predictor values are
imputed within each training fold. The two quantum noise conditions use exact
depolarizing density-matrix simulation (not finite-shot sampling).

To run the focused checks:

```powershell
python -m pip install pytest
python -m pytest test_experiment.py -q
```

Dataset citation: Quinlan, R. (1993). *Auto MPG*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5859H (CC BY 4.0).
