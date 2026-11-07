"""Screening tool: should I try a quantum kernel on this dataset? (report Section "The screening tool")

Runs before any SVM is trained. It uses training data only:

1. preprocess (X, y) like every training fold in the study (impute, one-hot, standardise,
   PCA to n_qubits, min-max to [0, pi]);
2. for RBF and each quantum map, pick the hyper-parameter with the highest centered
   kernel-target alignment (KTA) and keep that Gram matrix;
3. delta = KTA(best quantum map) - KTA(RBF);
4. recommend the quantum map if delta > tau, otherwise RBF.

tau is fixed in advance (DEFAULT_TAU below). It is the threshold the screening rule picks
when fitted on all 30 suite variants (moons, ad hoc, Heart Disease); recompute it with
`python scripts/screen.py --fit-tau`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import diagnostics as dg
from .data import Dataset, FoldPreprocessor
from .kernels import compute_gram
from .evaluate import expand_grid

DEFAULT_TAU = 0.0063  # fitted on all 30 suite variants; held-out fits gave 0.0056-0.0089
QUANTUM_MODELS = ("q_zz", "q_he", "q_product")
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
WARNING = ("tau was calibrated on moons, ad hoc and Heart Disease variants only; "
           "a large delta does not guarantee a quantum win (e.g. circles, where RBF is already at the ceiling).")


def _candidates(cfg: dict, name: str) -> list[dict]:
    m = dict(cfg["models"][name])
    grid = expand_grid(m.pop("grid", {}))
    return [{**m, **g} for g in grid]


def _best_by_alignment(cands: list[dict], X: np.ndarray, y: np.ndarray):
    best = None
    for c in cands:
        K = compute_gram(c, X)
        a = dg.centered_alignment(K, y)
        if best is None or a > best[0]:
            best = (a, c, K)
    return best


def _label(params: dict) -> str:
    keys = [k for k in ("bandwidth", "gamma_mult") if k in params]
    return ",".join(f"{k}={params[k]:g}" for k in keys)


def recommend_kernel(
    X, y=None, *, n_qubits: int = 4, tau: float = DEFAULT_TAU, categorical: list[str] | None = None,
    config: str | Path | dict | None = None, seed: int = 0,
) -> dict:
    """Recommend a quantum map or tuned RBF for a small labelled table. No SVM is trained.

    X: DataFrame / array of raw features, or a `Dataset` (then y is taken from it and its own
    preprocessing flags are respected, e.g. ad hoc data is used unscaled).
    Returns a dict with the recommendation, delta, tau, every kernel's best alignment, the
    effective-rank fraction of the chosen quantum Gram matrix, and a calibration warning.
    """
    if isinstance(config, dict):
        cfg = config
    else:
        cfg = yaml.safe_load(open(config or CONFIG))
    if isinstance(X, Dataset):
        ds = X
    else:
        if y is None:
            raise ValueError("y is required when X is not a Dataset")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, float),
                                                              columns=[f"x{i}" for i in range(np.shape(X)[1])])
        categorical = list(categorical or [])
        numeric = [c for c in df.columns if c not in categorical]
        ds = Dataset("input", df.reset_index(drop=True), np.asarray(y), numeric, categorical,
                     reduce=df.shape[1] > n_qubits)
    y = np.asarray(ds.y)
    if len(np.unique(y)) != 2:
        raise ValueError("recommend_kernel handles binary labels only")
    y = (y == np.unique(y)[1]).astype(int)

    n_comp = n_qubits if ds.reduce else ds.X.shape[1]
    Xp = FoldPreprocessor(ds, n_comp, seed).fit_transform(ds.X)

    kta_rbf, cfg_rbf, _ = _best_by_alignment(_candidates(cfg, "rbf"), Xp, y)
    alignments = {"rbf": kta_rbf}
    params = {"rbf": _label(cfg_rbf)}
    best_q = None
    for name in QUANTUM_MODELS:
        a, c, K = _best_by_alignment(_candidates(cfg, name), Xp, y)
        alignments[name] = a
        params[name] = _label(c)
        if best_q is None or a > best_q[1]:
            best_q = (name, a, K)
    q_name, q_kta, K_q = best_q
    delta = q_kta - kta_rbf
    use_q = delta > tau
    return {
        "recommendation": q_name if use_q else "rbf",
        "use_quantum": bool(use_q),
        "best_quantum_map": q_name,
        "delta": float(delta),
        "tau": float(tau),
        "alignment": alignments,
        "params": params,
        "rank_frac": dg.effective_rank(K_q) / len(K_q),
        "n_samples": len(y),
        "n_features_used": Xp.shape[1],
        "warning": WARNING,
    }
