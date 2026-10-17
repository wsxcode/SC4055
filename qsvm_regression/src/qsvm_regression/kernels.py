"""Shared classical and quantum kernel utilities for the benchmark."""

from __future__ import annotations

import math
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.model_selection import KFold

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



