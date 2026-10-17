"""Run the multi-dataset quantum-kernel regression study."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qsvm_regression.benchmark import RegressionDataset, TabularFoldPreprocessor, load_regression_dataset, select_fixed_sample
from qsvm_regression.experiment import (
    _compute_kernel,
    _environment_info,
    _git_info,
    _inner_splits,
    _kernel_diagnostics,
    _kernel_specs,
    _parameter_grid,
    temporal_year_split,
)

logger = logging.getLogger(__name__)


def _preprocess(data: RegressionDataset, train_idx: np.ndarray, eval_idx: np.ndarray, config: dict[str, Any], seed: int):
    prep = TabularFoldPreprocessor(
        data.numeric_columns, data.categorical_columns,
        int(config["preprocess"].get("n_components", 4)), seed,
    )
    return prep.fit_transform(data.X.iloc[train_idx], data.X.iloc[eval_idx])


def _tune(
    data: RegressionDataset, train_idx: np.ndarray, spec: dict[str, Any],
    config: dict[str, Any], seed: int,
) -> tuple[float, float, dict[str, float]]:
    grid = _parameter_grid(config)
    scores = {params: [] for params in grid}
    splits = _inner_splits(len(train_idx), int(config["cv"].get("inner_folds", 3)), seed)
    for inner_fold, (inner_train_local, inner_val_local) in enumerate(splits):
        inner_train = train_idx[inner_train_local]
        inner_val = train_idx[inner_val_local]
        X_train, X_val = _preprocess(data, inner_train, inner_val, config, seed + inner_fold)
        K_train, K_val, _ = _compute_kernel(spec, X_train, X_val, config, seed + inner_fold)
        y_scaler = StandardScaler().fit(data.y[inner_train].reshape(-1, 1))
        y_train = y_scaler.transform(data.y[inner_train].reshape(-1, 1)).ravel()
        for C, epsilon in grid:
            model = SVR(kernel="precomputed", C=C, epsilon=epsilon)
            model.fit(K_train, y_train)
            prediction = y_scaler.inverse_transform(model.predict(K_val).reshape(-1, 1)).ravel()
            scores[(C, epsilon)].append(mean_absolute_error(data.y[inner_val], prediction))
    means = {f"C={C:g},epsilon={epsilon:g}": float(np.mean(values)) for (C, epsilon), values in scores.items()}
    best = min(scores, key=lambda params: (np.mean(scores[params]), params[0], params[1]))
    return best[0], best[1], means


def _evaluate(
    data: RegressionDataset, train_idx: np.ndarray, test_idx: np.ndarray,
    protocol: str, fold: int, config: dict[str, Any], specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_seed = int(config["cv"].get("random_seed", 42)) + fold
    rows = []
    for spec_index, spec in enumerate(specs):
        seed = base_seed + spec_index * 100
        C, epsilon, inner_scores = _tune(data, train_idx, spec, config, seed)
        X_train, X_test = _preprocess(data, train_idx, test_idx, config, base_seed)
        K_train, K_test, kernel_time = _compute_kernel(spec, X_train, X_test, config, base_seed)
        y_scaler = StandardScaler().fit(data.y[train_idx].reshape(-1, 1))
        model = SVR(kernel="precomputed", C=C, epsilon=epsilon)
        model.fit(K_train, y_scaler.transform(data.y[train_idx].reshape(-1, 1)).ravel())
        prediction = y_scaler.inverse_transform(model.predict(K_test).reshape(-1, 1)).ravel()
        diagnostics = _kernel_diagnostics(K_train)
        rows.append({
            "dataset": data.name, "target": data.target, "protocol": protocol,
            "fold": fold, "model": spec["name"], "kernel_kind": spec["kind"],
            "feature_map": spec.get("feature_map"), "backend": spec.get("backend", "classical"),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "C": C, "epsilon_standardized_target": epsilon,
            "mae": float(mean_absolute_error(data.y[test_idx], prediction)),
            "rmse": float(np.sqrt(mean_squared_error(data.y[test_idx], prediction))),
            "r2": float(r2_score(data.y[test_idx], prediction)),
            "kernel_time_seconds": float(kernel_time),
            "kernel_psd": diagnostics["is_psd"],
            "kernel_min_eigenvalue": diagnostics["min_eigenvalue"],
            "kernel_effective_rank": diagnostics["effective_rank"],
            "inner_cv_mae": inner_scores,
        })
    return rows


def _learning_curve_rows(
    data: RegressionDataset, train_idx: np.ndarray, test_idx: np.ndarray,
    config: dict[str, Any], specs: list[dict[str, Any]], fold_rows: list[dict[str, Any]],
    smoke: bool,
) -> list[dict[str, Any]]:
    """Evaluate fixed representative kernels on growing subsets of one outer fold."""
    by_name = {spec["name"]: spec for spec in specs}
    selected = [name for name in ("linear", "rbf", "q_rot2dof_ideal", "q_belis_ideal") if name in by_name]
    fractions = [0.5, 1.0] if smoke else [0.2, 0.4, 0.6, 0.8, 1.0]
    result_rows = []
    for model_index, name in enumerate(selected):
        tuned = next(row for row in fold_rows if row["model"] == name)
        spec = by_name[name]
        rng = np.random.default_rng(int(config["cv"].get("random_seed", 42)) + 900 + model_index)
        train_order = rng.permutation(train_idx)
        for fraction_index, fraction in enumerate(fractions):
            n_train = max(2, int(np.ceil(len(train_idx) * fraction)))
            subset_idx = np.sort(train_order[:min(n_train, len(train_idx))])
            X_train, X_test = _preprocess(data, subset_idx, test_idx, config, 600 + fraction_index)
            K_train, K_test, seconds = _compute_kernel(spec, X_train, X_test, config, 700 + fraction_index)
            scaler = StandardScaler().fit(data.y[subset_idx].reshape(-1, 1))
            model = SVR(kernel="precomputed", C=float(tuned["C"]), epsilon=float(tuned["epsilon_standardized_target"]))
            model.fit(K_train, scaler.transform(data.y[subset_idx].reshape(-1, 1)).ravel())
            prediction = scaler.inverse_transform(model.predict(K_test).reshape(-1, 1)).ravel()
            result_rows.append({
                "dataset": data.name, "protocol": "first_outer_fold_fixed_parameters",
                "model": name, "train_fraction": fraction, "n_train": len(subset_idx),
                "n_test": len(test_idx), "mae": mean_absolute_error(data.y[test_idx], prediction),
                "rmse": np.sqrt(mean_squared_error(data.y[test_idx], prediction)),
                "r2": r2_score(data.y[test_idx], prediction), "kernel_time_seconds": seconds,
                "parameter_source": "nested-CV-selected on full first outer-fold training partition",
            })
    return result_rows


def _aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    return folds.groupby(["dataset", "protocol", "model"], as_index=False).agg(
        n_folds=("fold", "count"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        kernel_time_mean_seconds=("kernel_time_seconds", "mean"),
        effective_rank_mean=("kernel_effective_rank", "mean"),
        psd_fraction=("kernel_psd", "mean"),
    )


def _save_plot(aggregate: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(aggregate["dataset"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        part = aggregate[(aggregate.dataset == dataset) & (aggregate.protocol == "nested_cv")].sort_values("mae_mean")
        ax.barh(part.model, part.mae_mean, xerr=part.mae_std.fillna(0), color=["#2d6a4f" if m.startswith("q_") else "#457b9d" for m in part.model])
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_xlabel("MAE (lower is better)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Classical and quantum kernel SVR across regression datasets")
    fig.tight_layout()
    fig.savefig(output_dir / "mae_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(config: dict[str, Any], output_dir: str | Path | None = None, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(config["datasets"])
    seed = int(config["cv"].get("random_seed", 42))
    max_samples = int(config.get("sampling", {}).get("max_samples", 500))
    sample_seed = int(config.get("sampling", {}).get("random_seed", seed))
    outer_folds = int(config["cv"].get("outer_folds", 5))
    specs = _kernel_specs(config)
    if smoke:
        outer_folds = 2
        config = json.loads(json.dumps(config))
        config["cv"]["outer_folds"] = outer_folds
        config["cv"]["inner_folds"] = 2
        config["svr"]["C"] = [1.0]
        config["svr"]["epsilon"] = [0.5]
        smoke_models = {"linear", "q_rot2dof_ideal", "q_rot2dof_noisy"}
        specs = [spec for spec in specs if spec["name"] in smoke_models]

    output_path = Path(output_dir or config.get("output_dir", "results/multi_dataset_benchmark"))
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    dataset_metadata = []
    for name in names:
        logger.info("Loading %s", name)
        data = select_fixed_sample(load_regression_dataset(name), max_samples, sample_seed)
        if name == "auto_mpg" and data.temporal_year is not None:
            tr, te, years = temporal_year_split(data.temporal_year, int(config.get("temporal_holdout", {}).get("latest_year_count", 2)))
            all_rows.extend(_evaluate(data, tr, te, "latest_year_holdout", 0, config, specs))
            holdout = {"years": years.tolist(), "train_size": int(len(tr)), "test_size": int(len(te))}
        else:
            holdout = None
        splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        dataset_rows = []
        splits = list(splitter.split(data.X))
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            logger.info("%s nested CV fold %d/%d", name, fold, outer_folds)
            rows = _evaluate(data, train_idx, test_idx, "nested_cv", fold, config, specs)
            all_rows.extend(rows)
            if fold == 1:
                dataset_rows = rows
        learning_rows.extend(_learning_curve_rows(data, splits[0][0], splits[0][1], config, specs, dataset_rows, smoke))
        dataset_metadata.append({
            "name": name, "source": "UCI Machine Learning Repository", "source_id": data.source_id,
            "target": data.target, "citation": data.citation, "doi": data.doi,
            "license": data.license, "n_rows_source": data.source_rows,
            "n_rows_used": len(data.y), "n_features": int(data.X.shape[1]),
            "feature_columns": list(data.X.columns), "numeric_columns": data.numeric_columns,
            "categorical_columns": data.categorical_columns,
            "sample_row_indices": data.row_indices, "temporal_holdout": holdout,
        })

    folds = pd.DataFrame(all_rows)
    aggregate = _aggregate(folds)
    folds.to_csv(output_path / "fold_results.csv", index=False)
    aggregate.to_csv(output_path / "aggregate_results.csv", index=False)
    learning_curves = pd.DataFrame(learning_rows)
    learning_curves.to_csv(output_path / "learning_curve_results.csv", index=False)
    for name in names:
        subset = folds[folds.dataset == name]
        subset.to_csv(output_path / f"{name}_fold_results.csv", index=False)
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "reference_methodology": "Kakavand, Strohmeyer, and Schlotter (2026), arXiv:2604.18837; methods adapted for a distinct regression benchmark.",
        "config": config, "datasets": dataset_metadata,
        "sampling": {"max_samples": max_samples, "method": "fixed-seed target-quantile-stratified sample; all rows retained below cap", "seed": sample_seed},
        "target_scaling": "StandardScaler fitted within each training partition; metrics reported on the original target scale.",
        "environment": _environment_info(), "git": _git_info(), "smoke": smoke,
    }
    (output_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _save_plot(aggregate, output_path)
    return folds, aggregate


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs/benchmark.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a reduced end-to-end check on every configured dataset")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
    run_benchmark(load_config(args.config), args.output_dir, args.smoke)


if __name__ == "__main__":
    main()
