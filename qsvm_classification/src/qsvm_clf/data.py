"""Datasets and leakage-safe per-fold preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .quantum import statevectors

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
DATASETS = ("ad_hoc", "moons", "circles", "heart", "breast_cancer", "wine", "digits")
FIXED_DIM = {"moons": 2, "circles": 2}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric: list[str]
    categorical: list[str] = field(default_factory=list)
    reduce: bool = True      # PCA to n_components inside every training fold
    scale: str = "pi"        # "pi": min-max to [0, pi] (fit on train); "none": leave as is


# ---------------------------------------------------------------- ad hoc data
@lru_cache(maxsize=None)
def _oracle(n: int, oracle_seed: int = 0):
    """Fixed labelling function of the Havlicek et al. 'ad hoc' dataset.

    f(x) = <psi(x)| V^dag Z^(x n) V |psi(x)> with psi the 2-rep linear ZZ
    feature map and V a fixed (seeded) Haar-random unitary. Labels are the sign
    of f - median(f); points inside a margin around the median are rejected, so
    the classes are separable by a quantum kernel with a clear gap.
    """
    rng = np.random.default_rng(1234 + n + 7919 * oracle_seed)  # oracle_seed=0 reproduces the original dataset
    z = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    q, r = np.linalg.qr(z)
    V = q * (np.diag(r) / np.abs(np.diag(r)))
    parity = np.array([(-1) ** bin(i).count("1") for i in range(2**n)])
    obs = V.conj().T @ np.diag(parity) @ V
    pool = rng.uniform(0, 2 * np.pi, size=(20000, n))

    def f(x):
        s = statevectors(x, "zz", reps=2, entanglement="linear")
        return np.real(np.einsum("ni,ij,nj->n", s.conj(), obs, s))

    vals = f(pool)
    return f, float(np.median(vals)), float(np.std(vals))


def ad_hoc_data(n_samples: int, n_features: int, seed: int, gap_frac: float = 0.5, oracle_seed: int = 0):
    f, med, std = _oracle(n_features, oracle_seed)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    need = n_samples // 2
    counts = [0, 0]
    while min(counts) < need:
        x = rng.uniform(0, 2 * np.pi, size=(4096, n_features))
        v = f(x) - med
        keep = np.abs(v) > gap_frac * std
        for xi, vi in zip(x[keep], v[keep]):
            c = int(vi > 0)
            if counts[c] < need:
                xs.append(xi)
                ys.append(c)
                counts[c] += 1
    order = rng.permutation(len(xs))
    return np.array(xs)[order], np.array(ys)[order]


# ----------------------------------------------------------------- real data
def _load_heart() -> tuple[pd.DataFrame, np.ndarray]:
    path = CACHE_DIR / "heart_cleveland.csv"
    if not path.exists():
        from ucimlrepo import fetch_ucirepo

        d = fetch_ucirepo(id=45)  # Heart Disease (Cleveland subset has 303 rows)
        df = d.data.features.copy()
        df["target"] = (d.data.targets.iloc[:, 0].to_numpy() > 0).astype(int)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    return df.drop(columns="target"), df["target"].to_numpy()


def balanced_indices(y: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_class = min(n_samples // 2, *(int((y == c).sum()) for c in np.unique(y)))
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in np.unique(y)])
    return rng.permutation(idx)


def flip_labels(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Flip a random `fraction` of the labels (label noise), keeping the classes balanced in expectation."""
    if fraction <= 0:
        return y
    rng = np.random.default_rng(seed + 424242)
    y = y.copy()
    idx = rng.choice(len(y), int(round(fraction * len(y))), replace=False)
    y[idx] = 1 - y[idx]
    return y


def load_dataset(
    name: str, n_samples: int = 200, seed: int = 0, n_features: int = 4, *,
    oracle_seed: int = 0, flip: float = 0.0, noise: float | None = None,
) -> Dataset:
    """Load a balanced binary dataset of (at most) n_samples rows.

    n_features only matters for `ad_hoc`, which is generated directly in the
    requested dimension (real data is reduced with PCA inside each fold).
    Variants: `oracle_seed` selects another ad hoc generating circuit, `flip`
    adds label noise, `noise` sets the moons noise level (default 0.15).
    """
    if name == "ad_hoc":
        X, y = ad_hoc_data(n_samples, n_features, seed, oracle_seed=oracle_seed)
        cols = [f"x{i}" for i in range(n_features)]
        return Dataset(name, pd.DataFrame(X, columns=cols), flip_labels(y, flip, seed), cols, reduce=False, scale="none")
    if name in ("moons", "circles"):
        if name == "moons":
            X, y = make_moons(n_samples, noise=0.15 if noise is None else noise, random_state=seed)
        else:
            X, y = make_circles(n_samples, noise=0.08 if noise is None else noise, factor=0.5, random_state=seed)
        return Dataset(name, pd.DataFrame(X, columns=["x0", "x1"]), flip_labels(y, flip, seed), ["x0", "x1"], reduce=False)
    if name == "breast_cancer":
        b = load_breast_cancer(as_frame=True)
        X, y = b.data, b.target.to_numpy()
        numeric, categorical = list(X.columns), []
    elif name in ("wine", "digits"):
        # wine: cultivar 1 vs 2 (the two that overlap most; 71 + 48 rows, 13 features)
        # digits: 3 vs 8 (similar shapes; ~360 rows, 64 pixel features)
        b = load_wine(as_frame=True) if name == "wine" else load_digits(as_frame=True)
        pos, neg = (1, 2) if name == "wine" else (3, 8)
        keep = b.target.isin([pos, neg]).to_numpy()
        X = b.data[keep].reset_index(drop=True)
        y = (b.target[keep].to_numpy() == pos).astype(int)
        numeric, categorical = list(X.columns), []
    elif name == "heart":
        X, y = _load_heart()
        categorical = ["cp", "restecg", "slope", "thal"]
        numeric = [c for c in X.columns if c not in categorical]
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")
    idx = balanced_indices(y, n_samples, seed)
    return Dataset(name, X.iloc[idx].reset_index(drop=True), flip_labels(y[idx], flip, seed), numeric, categorical)


# -------------------------------------------------------------- preprocessing
class FoldPreprocessor:
    """Impute -> one-hot -> standardise -> PCA -> min-max to [0, pi].

    Every statistic is learned from the data passed to `fit_transform` only.
    """

    def __init__(self, ds: Dataset, n_components: int, seed: int = 0):
        self.ds, self.n_components, self.seed = ds, n_components, seed

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        ds = self.ds
        if ds.scale == "none" and not ds.reduce:
            return train[ds.numeric].to_numpy(float)
        self.num_imp = SimpleImputer(strategy="median").fit(train[ds.numeric])
        parts = [self.num_imp.transform(train[ds.numeric])]
        if ds.categorical:
            self.cat_imp = SimpleImputer(strategy="most_frequent").fit(train[ds.categorical])
            self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                self.cat_imp.transform(train[ds.categorical])
            )
            parts.append(self.enc.transform(self.cat_imp.transform(train[ds.categorical])))
        Z = np.column_stack(parts)
        self.std = StandardScaler().fit(Z)
        Z = self.std.transform(Z)
        self.pca = PCA(self.n_components, random_state=self.seed).fit(Z) if ds.reduce else None
        if self.pca is not None:
            Z = self.pca.transform(Z)
        self.mm = MinMaxScaler((0, np.pi)).fit(Z)
        return self.mm.transform(Z)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        ds = self.ds
        if ds.scale == "none" and not ds.reduce:
            return X[ds.numeric].to_numpy(float)
        parts = [self.num_imp.transform(X[ds.numeric])]
        if ds.categorical:
            parts.append(self.enc.transform(self.cat_imp.transform(X[ds.categorical])))
        Z = self.std.transform(np.column_stack(parts))
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return self.mm.transform(Z)
