"""Pre-training kernel diagnostics used to *predict* which kernel will win.

All quantities are computed on training data only (labels mapped to -1/+1).

* centered kernel-target alignment (Cortes et al. 2012)
* eigenvalue spectrum / effective rank
* geometric difference g(K_C || K_Q) and model complexity s_K(lambda)
  (Huang et al., Nat. Commun. 12, 2631, 2021)
* kernel concentration: variance of the off-diagonal entries
"""

from __future__ import annotations

import numpy as np


def _sym(K: np.ndarray) -> np.ndarray:
    return (K + K.T) / 2


def _pm1(y) -> np.ndarray:
    return np.where(np.asarray(y) > 0, 1.0, -1.0)


def centered_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """<Kc, Yc>_F / (||Kc||_F ||Yc||_F) with Yc the centered label kernel."""
    y = _pm1(y)
    n = len(y)
    H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H
    Yc = H @ np.outer(y, y) @ H
    denom = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / denom) if denom > 0 else 0.0


def normalise_trace(K: np.ndarray) -> np.ndarray:
    """Scale so that trace(K) = N (Huang et al. convention)."""
    return K * len(K) / np.trace(K)


def spectrum(K: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(_sym(K)))[::-1]


def effective_rank(K: np.ndarray) -> float:
    ev = np.clip(spectrum(K), 0, None)
    p = ev / ev.sum() if ev.sum() > 0 else ev
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0


def is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(spectrum(K).min() >= -tol)


def model_complexity(K: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> float:
    """s_K(lambda) = y^T (K + lambda I)^-1 y / N with K trace-normalised.

    Small s means the labels are easy to represent with this kernel (the
    generalisation bound of Huang et al. scales as sqrt(s / N)).
    """
    y = _pm1(y)
    Kn = normalise_trace(_sym(K))
    return float(y @ np.linalg.solve(Kn + lam * np.eye(len(y)), y) / len(y))


def geometric_difference(K_c: np.ndarray, K_q: np.ndarray, lam: float = 1e-3) -> float:
    """g(K_C || K_Q) = sqrt(|| sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf).

    Large g means K_Q can express functions that K_C cannot; g near 1 means
    the classical kernel can mimic the quantum one.
    """
    Kc, Kq = normalise_trace(_sym(K_c)), normalise_trace(_sym(K_q))
    w, V = np.linalg.eigh(Kq)
    sqrt_q = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sqrt_q @ np.linalg.solve(Kc + lam * np.eye(len(Kc)), sqrt_q)
    return float(np.sqrt(max(np.linalg.eigvalsh(_sym(M)).max(), 0.0)))


def concentration(K: np.ndarray) -> dict[str, float]:
    """Mean and variance of off-diagonal entries (kernel concentration)."""
    off = K[~np.eye(len(K), dtype=bool)]
    return {"offdiag_mean": float(off.mean()), "offdiag_var": float(off.var())}


def kernel_report(K: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> dict[str, float]:
    return {
        "kta": centered_alignment(K, y),
        "log_s": float(np.log(model_complexity(K, y, lam))),
        "eff_rank": effective_rank(K),
        **concentration(K),
    }
