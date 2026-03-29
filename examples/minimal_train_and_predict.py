#!/usr/bin/env python3
"""
Minimal FusionAge example using synthetic data only.

Uses LinearRegression via FusionAge.model (no PyTorch required).
For a DNN smoke test, install torch and call build_fusionage_dnn + train_dnn yourself.

Run from the repository root::

    python examples/minimal_train_and_predict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FusionAge import model as fm  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    n_train, n_val, p = 800, 200, 24
    beta = rng.standard_normal(p)
    X = rng.standard_normal((n_train + n_val, p)).astype(np.float64)
    y = X @ beta + rng.standard_normal(n_train + n_val) * 0.3

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    model = fm.build_linear_regression()
    model.fit(X_train, y_train)
    pred = fm.predict(model, X_val)
    mae = float(np.mean(np.abs(pred - y_val)))
    print(f"Hold-out MAE (dummy data, linear model): {mae:.6f}")
    print("OK — synthetic-data smoke test finished.")


if __name__ == "__main__":
    main()
