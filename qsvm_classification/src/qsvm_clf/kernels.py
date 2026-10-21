"""Gram-matrix computation (classical and quantum) with an on-disk cache."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from .quantum import fidelity_kernel, shot_noise_kernel

QUANTUM_KERNELS = ("zz", "product", "he")
CLASSICAL_KERNELS = ("linear", "rbf", "poly")


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _classical(cfg: dict, Xa: np.ndarray, Xb: np.ndarray | None) -> np.ndarray:
    ref = Xa if Xb is None else Xb  # Xb is always the training set when given
    gamma = float(cfg.get("gamma_mult", 1.0)) / (ref.shape[1] * ref.var())
    if cfg["kernel"] == "linear":
        return linear_kernel(Xa, Xb)
    if cfg["kernel"] == "rbf":
        return rbf_kernel(Xa, Xb, gamma=gamma)
    if cfg["kernel"] == "poly":
        return polynomial_kernel(Xa, Xb, degree=int(cfg.get("degree", 3)), gamma=gamma, coef0=1.0)
    raise ValueError(cfg["kernel"])


def _quantum(cfg: dict, Xa: np.ndarray, Xb: np.ndarray | None, key: str) -> np.ndarray:
    kw = dict(
        feature_map=cfg["kernel"],
        reps=int(cfg.get("reps", 2)),
        entanglement=cfg.get("entanglement", "linear"),
        bandwidth=float(cfg.get("bandwidth", 1.0)),
    )
    noise = cfg.get("noise") or {"mode": "ideal"}
    mode = noise.get("mode", "ideal")
    if mode == "aer":
        from .qiskit_ref import aer_noisy_kernel

        return aer_noisy_kernel(
            Xa, Xb, depolarizing_1q=float(noise.get("depolarizing_1q", 0.001)),
            depolarizing_2q=float(noise.get("depolarizing_2q", 0.01)), **kw,
        )
    K = fidelity_kernel(Xa, Xb, **kw)
    if mode == "shots":
        rng = np.random.default_rng(int(key[:12], 16))
        K = shot_noise_kernel(K, int(noise["shots"]), rng, symmetric=Xb is None)
    elif mode != "ideal":
        raise ValueError(f"Unknown noise mode {mode!r}")
    return K


def compute_gram(
    cfg: dict, Xa: np.ndarray, Xb: np.ndarray | None = None, cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Gram matrix K(Xa, Xb) (or K(Xa, Xa) if Xb is None).

    Disk caching is keyed on the kernel config and the exact input arrays, so
    a cached matrix can never be reused for different (e.g. re-fit) inputs.
    """
    key = _digest(cfg, Xa, Xb if Xb is not None else "self")
    # Only the Aer density-matrix kernels are slower than a disk round-trip;
    # caching the cheap ones just floods the disk with tiny files.
    expensive = (cfg.get("noise") or {}).get("mode") == "aer"
    path = Path(cache_dir) / f"{key}.npy" if cache_dir and expensive else None
    if path is not None and path.exists():
        return np.load(path)
    if cfg["kind"] == "classical":
        K = _classical(cfg, Xa, Xb)
    elif cfg["kind"] == "quantum":
        K = _quantum(cfg, Xa, Xb, key)
    else:
        raise ValueError(cfg["kind"])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, K)
    return K


def timed_gram(cfg, Xa, Xb=None, cache_dir=None) -> tuple[np.ndarray, float]:
    t = time.perf_counter()
    K = compute_gram(cfg, Xa, Xb, cache_dir)
    return K, time.perf_counter() - t
