#!/usr/bin/env python3
"""
Minimal FusionAge example using synthetic (dummy) data.

Demonstrates building and training the FusionAge DNN architecture on
random Gaussian features with a linear ground truth. No UK Biobank,
NHANES, or proprietary data is required.

Run from the repository root::

    python examples/minimal_train_and_predict.py

Requires: torch, numpy, tqdm, scikit-learn, xgboost (see environment.yml).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FusionAge import model as fm  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    n_train, n_val, p = 800, 200, 24
    beta = rng.standard_normal(p)
    X = rng.standard_normal((n_train + n_val, p)).astype(np.float32)
    y = (X @ beta + rng.standard_normal(n_train + n_val) * 0.3).astype(np.float32)

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    X_train_t, y_train_t, X_val_t, y_val_t = fm.prepare_tensors(
        X_train, y_train, X_val, y_val
    )

    print("Building FusionAge DNN (standard architecture)...")
    dnn = fm.build_fusionage_dnn(p, hidden_size=64, dropout=0.1)
    print(dnn)

    print("\nTraining DNN on synthetic data (40 epochs)...")
    dnn, hist, best_mse = fm.train_dnn(
        dnn,
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        n_epochs=40,
        batch_size=128,
        learning_rate=0.01,
        weight_decay=0.01,
        verbose=False,
    )

    pred = fm.predict(dnn, X_val)
    mae = float(np.mean(np.abs(pred - y_val)))
    print(f"\nBest validation MSE (training): {best_mse:.6f}")
    print(f"Hold-out MAE (same split):      {mae:.6f}")
    print("OK — synthetic-data smoke test finished.")


if __name__ == "__main__":
    main()
