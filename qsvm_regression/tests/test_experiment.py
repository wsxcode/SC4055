"""Focused checks for the Auto MPG regression experiment path."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qsvm_regression.experiment import (
    FoldPreprocessor,
    load_auto_mpg,
    temporal_year_split,
)


def test_loader_keeps_mpg_predictors_and_drops_car_identifier(monkeypatch):
    import ucimlrepo

    features = pd.DataFrame({
        "cylinders": [4, 6], "displacement": [90, 180], "horsepower": [60, np.nan],
        "weight": [1800, 2800], "acceleration": [15, 13], "model_year": [70, 71],
        "origin": [1, 2], "car_name": ["car-a", "car-b"],
    })
    targets = pd.DataFrame({"mpg": [30.0, 20.0]})
    monkeypatch.setattr(
        ucimlrepo, "fetch_ucirepo",
        lambda *, id: SimpleNamespace(data=SimpleNamespace(features=features, targets=targets)),
    )

    data = load_auto_mpg()

    assert data.X.shape == (2, 7)
    assert "car_name" not in data.X
    assert data.y.tolist() == [30.0, 20.0]
    assert data.model_year.tolist() == [70, 71]


def test_temporal_split_holds_out_latest_distinct_years():
    years = np.array([70, 70, 71, 72, 73, 73])
    train, test, held_out = temporal_year_split(years, latest_count=2)

    assert held_out.tolist() == [72, 73]
    assert years[train].max() < years[test].min()
    assert set(years[test]) == {72, 73}


def test_temporal_split_rejects_empty_training_period():
    with pytest.raises(ValueError, match="leave at least one"):
        temporal_year_split(np.array([1, 2]), latest_count=2)


def test_fold_preprocessor_fits_imputation_and_scaling_from_training_only():
    train = pd.DataFrame({
        "cylinders": [4, 6, 8, 4, 6],
        "displacement": [90, 150, 300, 100, 180],
        "horsepower": [60, np.nan, 120, 70, 90],
        "weight": [1800, 2500, 4000, 1900, 2800],
        "acceleration": [15, 14, 12, 16, 13],
        "model_year": [70, 71, 72, 73, 74],
        "origin": [1, 2, 3, 1, 2],
    })
    test = pd.DataFrame({
        "cylinders": [12],
        "displacement": [900],
        "horsepower": [np.nan],
        "weight": [9000],
        "acceleration": [2],
        "model_year": [99],
        "origin": [3],
    })
    processor = FoldPreprocessor(n_components=4, seed=42)
    X_train, X_test = processor.fit_transform(train, test)

    assert X_train.shape == (5, 4)
    assert X_test.shape == (1, 4)
    assert processor.numeric_imputer.statistics_[2] == pytest.approx(80.0)
    assert processor.scaler.mean_[0] == pytest.approx(
        processor.numeric_imputer.transform(train[processor_numeric_columns()])[:, 0].mean()
    )


def processor_numeric_columns():
    return ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
