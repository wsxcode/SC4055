"""Shared data and fold-local preprocessing for the regression benchmark."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .experiment import AutoMPGData, load_auto_mpg


@dataclass
class RegressionDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    target: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    citation: str
    doi: str
    license: str
    source_id: int
    source_rows: int = 0
    temporal_year: np.ndarray | None = None
    row_indices: list[int] | None = None


def _clean_columns(columns) -> list[str]:
    cleaned = []
    for column in columns:
        value = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        cleaned.append(value)
    return cleaned


def load_regression_dataset(name: str) -> RegressionDataset:
    """Load one of the four registered public regression tasks."""
    if name == "auto_mpg":
        data: AutoMPGData = load_auto_mpg()
        return RegressionDataset(
            name="auto_mpg", X=data.X, y=data.y, target="mpg",
            numeric_columns=["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"],
            categorical_columns=["origin"],
            citation="Quinlan, R. (1993). Auto MPG. UCI Machine Learning Repository.",
            doi="10.24432/C5859H", license="CC BY 4.0", source_id=9,
            source_rows=len(data.y), temporal_year=data.model_year,
        )

    from ucimlrepo import fetch_ucirepo

    specs: dict[str, dict[str, Any]] = {
        "concrete_strength": {
            "id": 165, "target": "concrete_compressive_strength", "citation": "Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.", "doi": "10.24432/C5PK67",
        },
        "energy_efficiency": {
            "id": 242, "target": "y1", "citation": "Tsanas, A. & Xifara, A. (2012). Energy Efficiency. UCI Machine Learning Repository.", "doi": "10.24432/C51307",
        },
        "bike_sharing": {
            "id": 275, "target": "cnt", "citation": "Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository.", "doi": "10.24432/C5W894",
        },
    }
    if name not in specs:
        raise ValueError(f"Unknown dataset {name!r}; choose from auto_mpg, {', '.join(specs)}")
    spec = specs[name]
    fetched = fetch_ucirepo(id=spec["id"])
    X_frame = fetched.data.features.copy()
    y_frame = fetched.data.targets.copy()
    X_frame.columns = _clean_columns(X_frame.columns)
    y_frame.columns = _clean_columns(y_frame.columns)
    target_col = spec["target"]
    if target_col not in y_frame.columns:
        raise ValueError(f"UCI dataset {spec['id']} is missing target {target_col!r}; found {list(y_frame.columns)}")

    # Bike Sharing publishes casual and registered counts alongside total count;
    # those are components of cnt and would leak the target.
    X_frame = X_frame.drop(columns=[c for c in ("casual", "registered", "instant", "dteday") if c in X_frame])
    y = pd.to_numeric(y_frame[target_col], errors="coerce").to_numpy(dtype=float)
    source_rows = len(y)
    keep = np.isfinite(y)
    X_frame = X_frame.loc[keep].reset_index(drop=True)
    y = y[keep]

    if name == "energy_efficiency":
        # Orientation and glazing-area distribution are discrete design choices.
        categorical = [c for c in ("x6", "x8") if c in X_frame]
    elif name == "bike_sharing":
        categorical = [c for c in ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit") if c in X_frame]
    else:
        categorical = []
    for col in X_frame:
        X_frame[col] = pd.to_numeric(X_frame[col], errors="coerce")
    numeric = [c for c in X_frame.columns if c not in categorical]
    return RegressionDataset(
        name=name, X=X_frame, y=y, target=target_col,
        numeric_columns=numeric, categorical_columns=categorical,
        citation=spec["citation"], doi=spec["doi"], license="CC BY 4.0", source_id=spec["id"],
        source_rows=source_rows,
    )


def select_fixed_sample(data: RegressionDataset, max_samples: int, seed: int) -> RegressionDataset:
    """Keep all rows under the cap; otherwise take a reproducible target-stratified sample."""
    if len(data.y) <= max_samples:
        data.row_indices = list(range(len(data.y)))
        return data
    # Quantile bins preserve the response range while avoiding class-specific sampling.
    n_bins = min(10, max(2, len(np.unique(data.y)) // 2))
    bins = pd.qcut(data.y, q=n_bins, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    unique_bins, counts = np.unique(bins, return_counts=True)
    quotas = np.floor(counts * max_samples / len(data.y)).astype(int)
    for i in range(max_samples - quotas.sum()):
        quotas[i % len(quotas)] += 1
    for value, quota in zip(unique_bins, quotas, strict=True):
        indices = np.flatnonzero(np.asarray(bins) == value)
        chosen.extend(rng.choice(indices, size=int(quota), replace=False).tolist())
    chosen = sorted(chosen)
    data.X = data.X.iloc[chosen].reset_index(drop=True)
    data.y = data.y[chosen]
    if data.temporal_year is not None:
        data.temporal_year = data.temporal_year[chosen]
    data.row_indices = chosen
    return data


class TabularFoldPreprocessor:
    """Impute, encode, scale, and reduce using only the training partition."""

    def __init__(self, numeric_columns: list[str], categorical_columns: list[str], n_components: int, seed: int):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.n_components = n_components
        self.seed = seed
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        parts_train, parts_test = [], []
        if self.numeric_columns:
            numeric_train = self.numeric_imputer.fit_transform(train[self.numeric_columns])
            numeric_test = self.numeric_imputer.transform(test[self.numeric_columns])
            parts_train.append(numeric_train)
            parts_test.append(numeric_test)
        if self.categorical_columns:
            cat_train = self.categorical_imputer.fit_transform(train[self.categorical_columns])
            cat_test = self.categorical_imputer.transform(test[self.categorical_columns])
            parts_train.append(self.encoder.fit_transform(cat_train))
            parts_test.append(self.encoder.transform(cat_test))
        X_train = self.scaler.fit_transform(np.column_stack(parts_train))
        X_test = self.scaler.transform(np.column_stack(parts_test))
        supported = min(X_train.shape[1], len(X_train))
        if self.n_components > supported:
            raise ValueError(f"Need {self.n_components} PCA components, but this fold supports {supported}")
        return self.pca.fit_transform(X_train), self.pca.transform(X_test)
