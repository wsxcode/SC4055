"""Fast exact statevector simulation of the quantum feature maps (NumPy).

Qubit q corresponds to axis q of the state tensor (qubit 0 is the most
significant bit of the basis index). One feature is encoded per qubit.

Feature maps
------------
zz       Havlicek et al. (2019) ZZFeatureMap: per repetition H, P(2x_i), and
         CX-P(2(pi-x_i)(pi-x_j))-CX on entangled pairs.
product  RY(x_i) on every qubit. Separable, so classically trivial; a control.
he       Hardware-efficient: per repetition RY(x_i) then a CZ ladder.
"""

from __future__ import annotations

import numpy as np

FEATURE_MAPS = ("zz", "product", "he")


def entangling_pairs(n: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "circular":
        pairs = [(i, i + 1) for i in range(n - 1)]
        return pairs + ([(n - 1, 0)] if n > 2 else [])
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"Unknown entanglement {entanglement!r}")


def _bits(n: int) -> np.ndarray:
    idx = np.arange(2**n)[:, None]
    return (idx >> (n - 1 - np.arange(n))[None, :]) & 1


def _hadamard_all(state: np.ndarray, n: int) -> np.ndarray:
    s = state.reshape(-1, *([2] * n))
    for ax in range(1, n + 1):
        a = np.take(s, 0, axis=ax)
        b = np.take(s, 1, axis=ax)
        s = np.stack([a + b, a - b], axis=ax) / np.sqrt(2.0)
    return s.reshape(state.shape)


def _ry_all(state: np.ndarray, angles: np.ndarray, n: int) -> np.ndarray:
    """Apply RY(angles[:, q]) to qubit q for every sample."""
    s = state.reshape(-1, *([2] * n))
    for q in range(n):
        shape = (-1,) + (1,) * (n - 1)
        c_ = np.cos(angles[:, q] / 2).reshape(shape)
        s_ = np.sin(angles[:, q] / 2).reshape(shape)
        a = np.take(s, 0, axis=q + 1)
        b = np.take(s, 1, axis=q + 1)
        s = np.stack([c_ * a - s_ * b, s_ * a + c_ * b], axis=q + 1)
    return s.reshape(state.shape)


def statevectors(
    X: np.ndarray,
    feature_map: str,
    reps: int = 2,
    entanglement: str = "linear",
    bandwidth: float = 1.0,
) -> np.ndarray:
    """Return an (N, 2**n) array of feature-map states for rows of X."""
    X = np.asarray(X, dtype=float) * bandwidth
    N, n = X.shape
    state = np.zeros((N, 2**n), dtype=complex)
    state[:, 0] = 1.0
    bits = _bits(n)
    pairs = entangling_pairs(n, entanglement)
    if feature_map == "zz":
        parity = {(i, j): (bits[:, i] ^ bits[:, j]) for i, j in pairs}
        for _ in range(reps):
            state = _hadamard_all(state, n)
            phase = 2.0 * X @ bits.T
            for i, j in pairs:
                theta = 2.0 * (np.pi - X[:, i]) * (np.pi - X[:, j])
                phase = phase + theta[:, None] * parity[(i, j)][None, :]
            state = state * np.exp(1j * phase)
    elif feature_map == "product":
        state = _ry_all(state, X, n)
    elif feature_map == "he":
        sign = np.ones(2**n)
        for i, j in pairs:
            sign = sign * (1 - 2 * (bits[:, i] & bits[:, j]))
        for _ in range(reps):
            state = _ry_all(state, X, n)
            state = state * sign[None, :]
    else:
        raise ValueError(f"Unknown feature map {feature_map!r}; choose from {FEATURE_MAPS}")
    return state


def fidelity_kernel(
    Xa: np.ndarray, Xb: np.ndarray | None, **map_kwargs
) -> np.ndarray:
    """Ideal kernel k(x, y) = |<psi(x)|psi(y)>|^2."""
    sa = statevectors(Xa, **map_kwargs)
    sb = sa if Xb is None else statevectors(Xb, **map_kwargs)
    return np.abs(sa @ sb.conj().T) ** 2


def shot_noise_kernel(K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool) -> np.ndarray:
    """Estimate each entry from `shots` samples of the compute-uncompute test.

    Each entry is Binomial(shots, k)/shots, which is exactly the statistics of
    measuring the all-zeros outcome of U(y)^dagger U(x)|0> with finite shots.
    """
    K = np.clip(K, 0.0, 1.0)
    est = rng.binomial(shots, K) / shots
    if symmetric:
        upper = np.triu(est, 1)
        est = upper + upper.T + np.eye(len(K))
    return est
