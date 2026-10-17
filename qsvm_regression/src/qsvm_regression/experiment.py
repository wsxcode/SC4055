"""Standalone, leakage-safe Auto MPG quantum-kernel SVR experiment."""

from __future__ import annotations

import json
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year",
]
ORIGIN_COLUMN = "origin"
CLASSICAL_KERNELS = ["linear", "rbf", "poly3"]
QUANTUM_MAPS = ["rot2dof", "belis"]


def _native_rx(circuit: QuantumCircuit, theta, qubit: int) -> None:
    """Use the benchmark's SX/RZ decomposition for a data rotation."""
    circuit.rz(np.pi / 2, qubit)
    circuit.sx(qubit)
    circuit.rz(theta, qubit)
    circuit.sx(qubit)
    circuit.rz(-np.pi / 2, qubit)


def _build_feature_map(name: str, n_features: int, reps: int) -> tuple[QuantumCircuit, list]:
    """Build the two small data-encoding circuits used in this experiment."""
    if name not in QUANTUM_MAPS:
        raise ValueError(f"Unknown feature map {name!r}; choose from {QUANTUM_MAPS}")
    n_qubits = math.ceil(n_features / 2)
    params = ParameterVector("x", n_features)
    circuit = QuantumCircuit(n_qubits, name=name)
    for _ in range(reps):
        for qubit in range(n_qubits):
            phi = params[2 * qubit]
            lam = params[2 * qubit + 1] if 2 * qubit + 1 < n_features else 0.0
            _native_rx(circuit, phi, qubit)
            circuit.rz(lam, qubit)
        if name == "belis":
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            for qubit in range(n_qubits):
                phi = params[2 * qubit]
                lam = params[2 * qubit + 1] if 2 * qubit + 1 < n_features else 0.0
                circuit.rz(phi, qubit)
                _native_rx(circuit, lam, qubit)
    return circuit, list(params)


def _parameterized_states(name: str, n_features: int, X: np.ndarray, reps: int) -> np.ndarray:
    circuit, params = _build_feature_map(name, n_features, reps)
    states = np.empty((len(X), 2 ** circuit.num_qubits), dtype=complex)
    for i, row in enumerate(X):
        bound = circuit.assign_parameters(dict(zip(params, row, strict=True)))
        states[i] = Statevector(bound).data
    return states


def _statevector_kernel(
    name: str, n_features: int, X: np.ndarray, Y: np.ndarray | None, reps: int,
) -> np.ndarray:
    states_x = _parameterized_states(name, n_features, X, reps)
    states_y = states_x if Y is None else _parameterized_states(name, n_features, Y, reps)
    return np.abs(states_x @ states_y.conj().T) ** 2


def _make_noisy_backend(noise: dict[str, Any], seed: int):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()
    error_1q = depolarizing_error(float(noise.get("depolarizing_1q", 0.001)), 1)
    error_2q = depolarizing_error(float(noise.get("depolarizing_2q", 0.01)), 2)
    for gate in ("sx", "rz", "x", "id"):
        noise_model.add_all_qubit_quantum_error(error_1q, gate)
    for gate in ("cx", "cz"):
        noise_model.add_all_qubit_quantum_error(error_2q, gate)
    return AerSimulator(method="density_matrix", noise_model=noise_model, seed_simulator=seed)


def _noisy_density_kernel(
    name: str,
    n_features: int,
    X: np.ndarray,
    Y: np.ndarray | None,
    reps: int,
    backend,
    seed: int,
) -> np.ndarray:
    """Compute a depolarizing-noise Hilbert-Schmidt kernel with Aer."""
    base, _ = _build_feature_map(name, n_features, reps)
    circuit = transpile(base, basis_gates=["sx", "rz", "cx", "id"], optimization_level=2)
    circuit.save_density_matrix()
    params = list(circuit.parameters)

    def embed(matrix: np.ndarray) -> np.ndarray:
        vectors = np.empty((len(matrix), (2 ** circuit.num_qubits) ** 2), dtype=complex)
        for start in range(0, len(matrix), 64):
            stop = min(start + 64, len(matrix))
            circuits = [
                circuit.assign_parameters(dict(zip(params, row, strict=True)))
                for row in matrix[start:stop]
            ]
            result = backend.run(circuits, shots=1, seed_simulator=seed + start).result()
            for local_idx in range(stop - start):
                rho = np.asarray(result.data(local_idx)["density_matrix"], dtype=complex)
                vectors[start + local_idx] = rho.ravel()
        return vectors

    states_x = embed(X)
    states_y = states_x if Y is None else embed(Y)
    return np.clip(np.real(states_x @ states_y.conj().T), 0.0, 1.0)


def _classical_kernel(name: str, X: np.ndarray, Y: np.ndarray | None) -> np.ndarray:
    if name == "linear":
        return linear_kernel(X, Y)
    if name == "rbf":
        gamma = 1.0 / (X.shape[1] * np.var(X))
        return rbf_kernel(X, Y, gamma=gamma)
    if name == "poly3":
        gamma = 1.0 / (X.shape[1] * np.var(X))
        return polynomial_kernel(X, Y, degree=3, gamma=gamma, coef0=0.0)
    raise ValueError(f"Unknown classical kernel {name!r}")


def _kernel_diagnostics(K: np.ndarray) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh((K + K.T) / 2)
    positive = np.maximum(eigenvalues, 0.0)
    total = positive.sum()
    probabilities = positive / total if total > 0 else positive
    probabilities = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if len(probabilities) else 0.0
    return {
        "is_psd": bool(eigenvalues.min() >= -1e-10),
        "min_eigenvalue": float(eigenvalues.min()),
        "effective_rank": effective_rank,
    }


def _environment_info() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "qiskit", "qiskit-aer", "ucimlrepo", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {"python_version": sys.version, "platform": platform.platform(), "packages": packages}


def _git_info() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit": commit}


@dataclass
class AutoMPGData:
    X: pd.DataFrame
    y: np.ndarray
    model_year: np.ndarray


def load_auto_mpg() -> AutoMPGData:
    """Fetch Auto MPG from UCI and return numeric predictors, target, and year."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=9)
    X = dataset.data.features.copy()
    y_frame = dataset.data.targets.copy()
    X.columns = [str(c).strip().lower().replace(" ", "_") for c in X.columns]
    y_frame.columns = [str(c).strip().lower().replace(" ", "_") for c in y_frame.columns]

    missing = set(NUMERIC_COLUMNS + [ORIGIN_COLUMN]) - set(X.columns)
    if missing:
        raise ValueError(f"Auto MPG is missing expected predictor columns: {sorted(missing)}")
    target_col = "mpg" if "mpg" in y_frame.columns else y_frame.columns[0]

    # The car name is a record identifier, not a predictor. Origin is nominal.
    predictors = X[NUMERIC_COLUMNS + [ORIGIN_COLUMN]].copy()
    for column in NUMERIC_COLUMNS + [ORIGIN_COLUMN]:
        predictors[column] = pd.to_numeric(predictors[column], errors="coerce")
    target = pd.to_numeric(y_frame[target_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(target)
    predictors = predictors.loc[keep].reset_index(drop=True)
    target = target[keep]
    years = predictors["model_year"].to_numpy(dtype=int)
    return AutoMPGData(X=predictors, y=target, model_year=years)


class FoldPreprocessor:
    """Fit median imputation, origin encoding, scaling, and PCA on train only."""

    def __init__(self, n_components: int = 4, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.origin_imputer = SimpleImputer(strategy="most_frequent")
        self.origin_encoder = OneHotEncoder(
            categories=[[1, 2, 3]], handle_unknown="ignore", sparse_output=False,
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)

    def fit_transform(
        self, train: pd.DataFrame, test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric_train = self.numeric_imputer.fit_transform(train[NUMERIC_COLUMNS])
        numeric_test = self.numeric_imputer.transform(test[NUMERIC_COLUMNS])
        origin_train = self.origin_encoder.fit_transform(
            self.origin_imputer.fit_transform(train[[ORIGIN_COLUMN]])
        )
        origin_test = self.origin_encoder.transform(
            self.origin_imputer.transform(test[[ORIGIN_COLUMN]])
        )
        X_train = np.column_stack([numeric_train, origin_train])
        X_test = np.column_stack([numeric_test, origin_test])
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        effective_components = min(self.n_components, X_train.shape[1], len(X_train))
        if effective_components != self.n_components:
            raise ValueError(
                f"Need {self.n_components} PCA components, but fold supports {effective_components}"
            )
        return self.pca.fit_transform(X_train), self.pca.transform(X_test)


def _kernel_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [{"name": name, "kind": "classical", "kernel": name}
             for name in config["kernels"].get("classical", CLASSICAL_KERNELS)]
    noise = config["kernels"].get("noisy_simulation", {})
    for feature_map in config["kernels"].get("quantum_feature_maps", QUANTUM_MAPS):
        specs.append({
            "name": f"q_{feature_map}_ideal", "kind": "quantum",
            "feature_map": feature_map, "backend": "ideal",
        })
        specs.append({
            "name": f"q_{feature_map}_noisy", "kind": "quantum",
            "feature_map": feature_map, "backend": "noisy",
            "noise": noise,
        })
    return specs


def _compute_kernel(
    spec: dict[str, Any], X_train: np.ndarray, X_eval: np.ndarray,
    config: dict[str, Any], seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    if spec["kind"] == "classical":
        K_train = _classical_kernel(spec["kernel"], X_train, None)
        K_eval = _classical_kernel(spec["kernel"], X_eval, X_train)
    else:
        n_features = X_train.shape[1]
        reps = config["kernels"].get("quantum_reps", 1)
        if spec["backend"] == "ideal":
            K_train = _statevector_kernel(spec["feature_map"], n_features, X_train, None, reps)
            K_eval = _statevector_kernel(spec["feature_map"], n_features, X_eval, X_train, reps)
        else:
            noise = spec["noise"]
            backend = _make_noisy_backend(noise, seed)
            K_train = _noisy_density_kernel(
                spec["feature_map"], n_features, X_train, None, reps, backend, seed,
            )
            K_eval = _noisy_density_kernel(
                spec["feature_map"], n_features, X_eval, X_train, reps, backend, seed,
            )
    return K_train, K_eval, time.perf_counter() - start


def _parameter_grid(config: dict[str, Any]) -> list[tuple[float, float]]:
    svr_cfg = config["svr"]
    return [(float(c), float(eps)) for c in svr_cfg["C"] for eps in svr_cfg["epsilon"]]


def _inner_splits(n: int, folds: int, seed: int):
    folds = min(folds, n)
    if folds < 2:
        raise ValueError("At least two training examples are required for inner CV")
    return list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(n)))


def temporal_year_split(years: np.ndarray, latest_count: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test indices, holding out the latest distinct years."""
    values = np.asarray(years)
    unique_years = np.sort(np.unique(values))
    if latest_count < 1 or latest_count >= len(unique_years):
        raise ValueError("latest_count must leave at least one distinct year for training")
    held_out_years = unique_years[-latest_count:]
    test_idx = np.flatnonzero(np.isin(values, held_out_years))
    train_idx = np.flatnonzero(~np.isin(values, held_out_years))
    return train_idx, test_idx, held_out_years


def _tune_parameters(
    X: pd.DataFrame,
    y: np.ndarray,
    spec: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[float, float, dict[str, float]]:
    """Select C/epsilon by inner-CV MAE with fold-local preprocessing."""
    n_components = int(config["preprocess"].get("n_components", 4))
    inner_folds = int(config["cv"].get("inner_folds", 3))
    scores = {params: [] for params in _parameter_grid(config)}
    splits = _inner_splits(len(y), inner_folds, seed)
    for inner_idx, (train_idx, val_idx) in enumerate(splits):
        processor = FoldPreprocessor(n_components=n_components, seed=seed)
        X_train, X_val = processor.fit_transform(X.iloc[train_idx], X.iloc[val_idx])
        K_train, K_val, _ = _compute_kernel(
            spec, X_train, X_val, config, seed + inner_idx,
        )
        for C, epsilon in scores:
            model = SVR(kernel="precomputed", C=C, epsilon=epsilon)
            model.fit(K_train, y[train_idx])
            pred = model.predict(K_val)
            scores[(C, epsilon)].append(mean_absolute_error(y[val_idx], pred))
    means = {f"C={C:g},epsilon={eps:g}": float(np.mean(values))
             for (C, eps), values in scores.items()}
    best = min(scores, key=lambda p: (np.mean(scores[p]), p[0], p[1]))
    return best[0], best[1], means


def _evaluate_partition(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    protocol: str,
    fold: int,
    config: dict[str, Any],
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seed = int(config["cv"].get("random_seed", 42)) + fold
    output = []
    for spec_idx, spec in enumerate(specs):
        C, epsilon, inner_scores = _tune_parameters(
            X.iloc[train_idx].reset_index(drop=True), y[train_idx], spec,
            config, seed + 100 * spec_idx,
        )
        processor = FoldPreprocessor(
            n_components=int(config["preprocess"].get("n_components", 4)),
            seed=seed,
        )
        X_train, X_test = processor.fit_transform(X.iloc[train_idx], X.iloc[test_idx])
        K_train, K_test, kernel_time = _compute_kernel(
            spec, X_train, X_test, config, seed,
        )
        model = SVR(kernel="precomputed", C=C, epsilon=epsilon)
        model.fit(K_train, y[train_idx])
        prediction = model.predict(K_test)
        diagnostics = _kernel_diagnostics(K_train)
        output.append({
            "protocol": protocol,
            "fold": fold,
            "model": spec["name"],
            "kernel_kind": spec["kind"],
            "feature_map": spec.get("feature_map"),
            "backend": spec.get("backend", "classical"),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "C": C,
            "epsilon": epsilon,
            "mae": float(mean_absolute_error(y[test_idx], prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
            "r2": float(r2_score(y[test_idx], prediction)),
            "kernel_time_seconds": float(kernel_time),
            "kernel_psd": diagnostics["is_psd"],
            "kernel_min_eigenvalue": diagnostics["min_eigenvalue"],
            "kernel_effective_rank": diagnostics["effective_rank"],
            "inner_cv_mae": inner_scores,
        })
    return output


def run_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested CV and later-year holdout and save reproducible artifacts."""
    data = load_auto_mpg()
    seed = int(config["cv"].get("random_seed", 42))
    outer_folds = int(config["cv"].get("outer_folds", 5))
    specs = _kernel_specs(config)
    if smoke:
        outer_folds = 2
        smoke_names = {"linear", "q_rot2dof_ideal", "q_rot2dof_noisy"}
        specs = [spec for spec in specs if spec["name"] in smoke_names]
        config = json.loads(json.dumps(config))
        config["cv"]["outer_folds"] = outer_folds
        config["cv"]["inner_folds"] = 2
        config["svr"]["C"] = [1.0]
        config["svr"]["epsilon"] = [0.5]

    all_rows: list[dict[str, Any]] = []
    splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data.X), start=1):
        logger.info("Random CV fold %d/%d", fold, outer_folds)
        all_rows.extend(_evaluate_partition(
            data.X, data.y, train_idx, test_idx, "nested_cv", fold, config, specs,
        ))

    latest_count = int(config.get("temporal_holdout", {}).get("latest_year_count", 2))
    temporal_train, temporal_test, latest_years = temporal_year_split(data.model_year, latest_count)
    logger.info("Temporal holdout years=%s", latest_years.tolist())
    all_rows.extend(_evaluate_partition(
        data.X, data.y, temporal_train, temporal_test, "latest_year_holdout", 0, config, specs,
    ))

    fold_results = pd.DataFrame(all_rows)
    aggregate = (
        fold_results.groupby(["protocol", "model"], as_index=False)
        .agg(
            n_folds=("fold", "count"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"), r2_std=("r2", "std"),
            kernel_time_mean_seconds=("kernel_time_seconds", "mean"),
            effective_rank_mean=("kernel_effective_rank", "mean"),
            psd_fraction=("kernel_psd", "mean"),
        )
    )
    configured_output = Path(output_dir or config.get("output_dir", "results/auto_mpg_regression"))
    project_dir = Path(__file__).resolve().parents[2]
    out_dir = configured_output if configured_output.is_absolute() else project_dir / configured_output
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_dir / "fold_results.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_results.csv", index=False)
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "method_reference": (
            "Kakavand, S., Strohmeyer, C., and Schlotter, M. (2026). "
            "Benchmarking Quantum Kernel Support Vector Machines Against Classical "
            "Baselines on Tabular Data: A Rigorous Empirical Study with Hardware Validation. "
            "arXiv:2604.18837. https://arxiv.org/abs/2604.18837"
        ),
        "config": config,
        "dataset": {
            "name": "Auto MPG",
            "uci_id": 9,
            "citation": "Quinlan, R. (1993). Auto MPG. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H",
            "license": "CC BY 4.0",
            "n_samples": int(len(data.y)),
            "n_features_before_encoding": 7,
            "feature_columns": list(data.X.columns),
            "target": "mpg",
            "excluded_identifier": "car_name",
            "latest_holdout_years": latest_years.tolist(),
            "temporal_train_size": int(len(temporal_train)),
            "temporal_test_size": int(len(temporal_test)),
        },
        "environment": _environment_info(),
        "git": _git_info(),
        "smoke": smoke,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    _save_plots(aggregate, out_dir)
    return fold_results, aggregate


def _save_plots(aggregate: pd.DataFrame, out_dir: Path) -> None:
    protocols = list(aggregate["protocol"].unique())
    fig, axes = plt.subplots(1, len(protocols), figsize=(7 * len(protocols), 5), squeeze=False)
    for ax, protocol in zip(axes[0], protocols, strict=False):
        subset = aggregate[aggregate["protocol"] == protocol].sort_values("mae_mean")
        yerr = subset["mae_std"].fillna(0).to_numpy()
        ax.barh(subset["model"], subset["mae_mean"], xerr=yerr, color=[
            "#2d6a4f" if name.startswith("q_") else "#457b9d" for name in subset["model"]
        ])
        ax.set_title(protocol.replace("_", " ").title())
        ax.set_xlabel("MAE (MPG; lower is better)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Auto MPG kernel SVR comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "mae_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parents[2]
    parser.add_argument("--config", default=project_dir / "configs/auto_mpg.yaml", help="Experiment YAML file")
    parser.add_argument("--output-dir", default=None, help="Override configured output directory")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced CV smoke run")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
    run_experiment(load_config(args.config), output_dir=args.output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
