"""Repeated stratified nested CV for classical and quantum kernel SVMs.

Fairness rules enforced here
* every model gets the same folds, the same preprocessing (fit on the training
  partition only, including inside the inner tuning folds);
* every model is tuned by the same inner CV: C plus ONE kernel hyper-parameter
  family (RBF: gamma, polynomial: gamma x degree, quantum: bandwidth);
* diagnostics are computed on the outer *training* partition only.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from . import diagnostics as dg
from .data import Dataset, FoldPreprocessor
from .kernels import compute_gram, timed_gram


@dataclass
class ModelSpec:
    name: str
    kind: str                       # "classical" | "quantum"
    kernel: str                     # linear/rbf/poly | zz/product/he
    grid: list[dict] = field(default_factory=lambda: [{}])   # kernel hyper-parameter candidates
    fixed: dict = field(default_factory=dict)                # reps, entanglement, noise, ...
    C: tuple = (0.1, 1.0, 10.0, 100.0)

    def kernel_configs(self) -> list[dict]:
        return [{"kind": self.kind, "kernel": self.kernel, **self.fixed, **g} for g in self.grid]


def expand_grid(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def model_from_config(name: str, cfg: dict, C: tuple) -> ModelSpec:
    cfg = dict(cfg)
    return ModelSpec(
        name=name, kind=cfg.pop("kind"), kernel=cfg.pop("kernel"),
        grid=expand_grid(cfg.pop("grid", {})), C=tuple(C), fixed=cfg,
    )


def _stratified_subsample(idx: np.ndarray, y: np.ndarray, m: int, rng) -> np.ndarray:
    per = m // 2
    keep = [rng.choice(idx[y == c], min(per, int((y == c).sum())), replace=False) for c in np.unique(y)]
    return np.sort(np.concatenate(keep))


def _fmt(params: dict) -> str:
    return ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in params.items())


def _tune(models, ds, tr, n_components, cv, seed, cache_dir):
    """Inner CV -> best (grid index, C) per model."""
    scores = {m.name: {} for m in models}
    inner = StratifiedKFold(cv["inner_folds"], shuffle=True, random_state=seed)
    for k, (itr, iva) in enumerate(inner.split(tr, ds.y[tr])):
        a, b = tr[itr], tr[iva]
        pre = FoldPreprocessor(ds, n_components, seed + k)
        Xa = pre.fit_transform(ds.X.iloc[a])
        Xb = pre.transform(ds.X.iloc[b])
        for m in models:
            for gi, kcfg in enumerate(m.kernel_configs()):
                Ka = compute_gram(kcfg, Xa, None, cache_dir)
                Kb = compute_gram(kcfg, Xb, Xa, cache_dir)
                for C in m.C:
                    clf = SVC(kernel="precomputed", C=C).fit(Ka, ds.y[a])
                    scores[m.name].setdefault((gi, C), []).append(accuracy_score(ds.y[b], clf.predict(Kb)))
    best = {}
    for m in models:
        s = scores[m.name]
        best[m.name] = min(s, key=lambda key: (-np.mean(s[key]), key[1], key[0]))  # ties -> smaller C
    return best


def run_fold(models, ds, tr, te, n_components, cv, seed, cache_dir=None, diagnostics=False, lam=1e-3):
    best = _tune(models, ds, tr, n_components, cv, seed, cache_dir)
    pre = FoldPreprocessor(ds, n_components, seed)
    Xtr = pre.fit_transform(ds.X.iloc[tr])
    Xte = pre.transform(ds.X.iloc[te])
    ytr, yte = ds.y[tr], ds.y[te]
    rows, diag_K = [], {}
    for m in models:
        gi, C = best[m.name]
        kcfg = m.kernel_configs()[gi]
        t0 = time.perf_counter()
        Ktr = compute_gram(kcfg, Xtr, None, cache_dir)
        Kte = compute_gram(kcfg, Xte, Xtr, cache_dir)
        kernel_s = time.perf_counter() - t0
        clf = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
        pred, dec = clf.predict(Kte), clf.decision_function(Kte)
        row = {
            "model": m.name, "kind": m.kind, "kernel": m.kernel, "best_C": C,
            "best_kernel_params": _fmt(m.grid[gi]), "n_train": len(tr), "n_test": len(te),
            "acc": accuracy_score(yte, pred), "f1": f1_score(yte, pred),
            "auc": roc_auc_score(yte, dec), "train_acc": clf.score(Ktr, ytr),
            "kernel_seconds": kernel_s,
        }
        if diagnostics:
            # kernel hyper-parameter chosen by training-set alignment alone (no SVM, no CV)
            grams = [compute_gram(c, Xtr, None, cache_dir) for c in m.kernel_configs()]
            ktas = [dg.centered_alignment(K, ytr) for K in grams]
            j = int(np.argmax(ktas))
            row.update({f"diag_{k}": v for k, v in dg.kernel_report(grams[j], ytr, lam).items()})
            row["diag_params"] = _fmt(m.grid[j])
            row["diag_psd"] = dg.is_psd(grams[j], 1e-6)
            diag_K[m.name] = grams[j]
        rows.append(row)
    if diagnostics and "rbf" in diag_K:
        for row in rows:
            if row["kind"] == "quantum":
                K_q = diag_K[row["model"]]
                row["diag_g_cq"] = dg.geometric_difference(diag_K["rbf"], K_q, lam)
    return rows


def run_cv(
    ds: Dataset, n_components: int, models: list[ModelSpec], cv: dict, *, seed: int = 0,
    train_size: int | None = None, cache_dir=None, diagnostics: bool = False, lam: float = 1e-3,
    meta: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for rep in range(cv["repeats"]):
        outer = StratifiedKFold(cv["outer_folds"], shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(outer.split(ds.X, ds.y)):
            rng = np.random.default_rng(seed * 10007 + rep * 101 + fold)
            if train_size is not None and train_size < len(tr):
                tr = _stratified_subsample(tr, ds.y[tr], train_size, rng)
            fold_rows = run_fold(
                models, ds, tr, te, n_components, cv, seed * 10007 + rep * 101 + fold,
                cache_dir, diagnostics, lam,
            )
            for r in fold_rows:
                rows.append({"dataset": ds.name, "n_qubits": n_components, "repeat": rep, "fold": fold,
                             **(meta or {}), **r})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- statistics
def corrected_resampled_ttest(diff: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected paired t-test for repeated CV.

    `diff` holds per-fold score differences (all repeats x folds). The usual
    t-test over-states significance because folds share training data.
    """
    from scipy import stats

    k = len(diff)
    var = np.var(diff, ddof=1)
    if k < 2 or var == 0:
        return (0.0, 1.0) if np.mean(diff) == 0 else (np.inf, 0.0)
    t = np.mean(diff) / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))
