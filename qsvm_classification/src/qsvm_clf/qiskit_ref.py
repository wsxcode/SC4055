"""Qiskit reference circuits: used to validate the NumPy simulator and to run
the Aer depolarizing-noise kernel."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector

from .quantum import entangling_pairs


def build_circuit(
    n: int, feature_map: str, reps: int = 2, entanglement: str = "linear"
) -> tuple[QuantumCircuit, ParameterVector]:
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    pairs = entangling_pairs(n, entanglement)
    if feature_map == "zz":
        for _ in range(reps):
            qc.h(range(n))
            for i in range(n):
                qc.p(2 * x[i], i)
            for i, j in pairs:
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    elif feature_map == "product":
        for i in range(n):
            qc.ry(x[i], i)
    elif feature_map == "he":
        for _ in range(reps):
            for i in range(n):
                qc.ry(x[i], i)
            for i, j in pairs:
                qc.cz(i, j)
    else:
        raise ValueError(feature_map)
    return qc, x


def qiskit_statevectors(X: np.ndarray, bandwidth: float = 1.0, **kw) -> np.ndarray:
    """Qiskit statevectors, reordered to the NumPy simulator's bit order."""
    from qiskit.quantum_info import Statevector

    X = np.asarray(X, float) * bandwidth
    n = X.shape[1]
    qc, params = build_circuit(n, **kw)
    out = np.empty((len(X), 2**n), complex)
    # Qiskit is little-endian (qubit 0 = least significant bit); ours is big-endian.
    perm = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)])
    for k, row in enumerate(X):
        sv = Statevector(qc.assign_parameters(dict(zip(params, row))))
        out[k] = sv.data[perm]
    return out


def make_noisy_backend(depolarizing_1q: float, depolarizing_2q: float, seed: int = 0):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    e1 = depolarizing_error(depolarizing_1q, 1)
    e2 = depolarizing_error(depolarizing_2q, 2)
    for gate in ("sx", "rz", "x", "id"):
        nm.add_all_qubit_quantum_error(e1, gate)
    nm.add_all_qubit_quantum_error(e2, "cx")
    return AerSimulator(method="density_matrix", noise_model=nm, seed_simulator=seed)


def noisy_density_vectors(
    X: np.ndarray, backend, feature_map: str, reps: int, entanglement: str, bandwidth: float
) -> np.ndarray:
    """Vectorised noisy density matrices rho(x) from Aer."""
    X = np.asarray(X, float) * bandwidth
    n = X.shape[1]
    base, params = build_circuit(n, feature_map, reps, entanglement)
    circ = transpile(base, basis_gates=["sx", "rz", "cx", "id"], optimization_level=1)
    circ.save_density_matrix()
    plist = list(circ.parameters)
    vecs = np.empty((len(X), (2**n) ** 2), complex)
    for s in range(0, len(X), 64):
        batch = [circ.assign_parameters(dict(zip(plist, row))) for row in X[s : s + 64]]
        res = backend.run(batch, shots=1).result()
        for k in range(len(batch)):
            vecs[s + k] = np.asarray(res.data(k)["density_matrix"], complex).ravel()
    return vecs


def aer_noisy_kernel(Xa, Xb, *, feature_map, reps, entanglement, bandwidth, depolarizing_1q, depolarizing_2q, seed=0):
    """Hilbert-Schmidt overlap Tr(rho(x) rho(y)) of the noisy feature states."""
    backend = make_noisy_backend(depolarizing_1q, depolarizing_2q, seed)
    kw = dict(feature_map=feature_map, reps=reps, entanglement=entanglement, bandwidth=bandwidth)
    va = noisy_density_vectors(Xa, backend, **kw)
    vb = va if Xb is None else noisy_density_vectors(Xb, backend, **kw)
    return np.clip(np.real(va @ vb.conj().T), 0.0, 1.0)
