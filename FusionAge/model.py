"""
FusionAge model architectures and training utilities.

Defines all regression model architectures used for training FusionAge aging
clocks, along with training and prediction utilities.

Models:
    Scikit-learn based:
        - Linear Regression
        - LassoCV (L1 regularization with built-in cross-validation)
        - ElasticNetCV (L1/L2 regularization with built-in cross-validation)

    XGBoost:
        - XGBRegressor with GridSearchCV for hyperparameter tuning

    PyTorch DNN (two variants):
        - Standard: 3 hidden blocks -> 32 -> 16 -> 8 -> 1
        - Deep: 3 hidden blocks -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1

Reference:
    Chen et al., "FusionAge framework for multimodal machine learning-based
    aging clocks uncovers cardiorespiratory fitness as a major driver of aging
    and inflammatory drivers of aging in response to spaceflight."
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

# DNN defaults (as reported in the paper)
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_N_EPOCHS = 300
DEFAULT_BATCH_SIZE = 1000

# LassoCV / ElasticNetCV defaults
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
DEFAULT_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
DEFAULT_MAX_ITER = 50000

# XGBoost grid search defaults
DEFAULT_XGB_PARAM_GRID = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [1, 3, 5],
    'n_estimators': [1, 501, 1001],
    'colsample_bytree': [0.5, 0.75, 1.0],
    'subsample': [0.5, 0.75, 1.0],
}


# ===========================================================================
# Scikit-learn models
# ===========================================================================

def build_linear_regression():
    """Build a standard linear regression model.

    Returns
    -------
    LinearRegression
        An unfitted scikit-learn LinearRegression estimator.
    """
    return LinearRegression()


def build_lasso(X_cv=None, y_cv=None,
                alphas=None, cv=10, n_jobs=2, max_iter=DEFAULT_MAX_ITER):
    """Build a LassoCV model with built-in cross-validation for alpha selection.

    If ``X_cv`` and ``y_cv`` are provided, the model is fit on the
    cross-validation set to select the optimal alpha before being passed
    to the main cross-validation loop in ``training.py``.

    Parameters
    ----------
    X_cv : array-like, optional
        Cross-validation feature matrix for alpha selection.
    y_cv : array-like, optional
        Cross-validation target vector for alpha selection.
    alphas : list of float, optional
        Candidate regularization strengths
        (default: [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]).
    cv : int, optional
        Number of CV folds for alpha selection (default: 10).
    n_jobs : int, optional
        Number of parallel jobs (default: 2).
    max_iter : int, optional
        Maximum solver iterations (default: 50000).

    Returns
    -------
    LassoCV
        A LassoCV estimator (fitted if X_cv/y_cv provided, unfitted otherwise).
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    regressor = LassoCV(cv=cv, n_jobs=n_jobs, alphas=alphas, max_iter=max_iter)

    if X_cv is not None and y_cv is not None:
        regressor.fit(X_cv, y_cv)

    return regressor


def build_elasticnet(X_cv=None, y_cv=None,
                     alphas=None, l1_ratios=None,
                     cv=10, n_jobs=2, max_iter=DEFAULT_MAX_ITER):
    """Build an ElasticNetCV model with built-in cross-validation.

    If ``X_cv`` and ``y_cv`` are provided, the model is fit on the
    cross-validation set to select the optimal alpha and l1_ratio before
    being passed to the main cross-validation loop in ``training.py``.

    Parameters
    ----------
    X_cv : array-like, optional
        Cross-validation feature matrix for hyperparameter selection.
    y_cv : array-like, optional
        Cross-validation target vector for hyperparameter selection.
    alphas : list of float, optional
        Candidate regularization strengths
        (default: [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]).
    l1_ratios : list of float, optional
        Candidate L1/L2 mixing parameters
        (default: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]).
    cv : int, optional
        Number of CV folds for hyperparameter selection (default: 10).
    n_jobs : int, optional
        Number of parallel jobs (default: 2).
    max_iter : int, optional
        Maximum solver iterations (default: 50000).

    Returns
    -------
    ElasticNetCV
        An ElasticNetCV estimator (fitted if X_cv/y_cv provided, unfitted
        otherwise).
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIOS

    regressor = ElasticNetCV(
        cv=cv, n_jobs=n_jobs,
        alphas=alphas, l1_ratio=l1_ratios,
        max_iter=max_iter,
    )

    if X_cv is not None and y_cv is not None:
        regressor.fit(X_cv, y_cv)

    return regressor


# ===========================================================================
# XGBoost
# ===========================================================================

def build_xgboost(X_cv=None, y_cv=None,
                  param_grid=None, cv=10, n_jobs=-3,
                  n_samples_grid_search=None,
                  feature_names=None):
    """Build an XGBRegressor with optional GridSearchCV hyperparameter tuning.

    If ``X_cv`` and ``y_cv`` are provided, a grid search is performed to
    find the best hyperparameters, and an XGBRegressor instantiated with
    those parameters is returned.  Otherwise, a default XGBRegressor is
    returned.

    Parameters
    ----------
    X_cv : array-like, optional
        Cross-validation feature matrix for grid search.
    y_cv : array-like, optional
        Cross-validation target vector for grid search.
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV.  Defaults to::

            {
                'learning_rate': [0.01, 0.1],
                'max_depth': [1, 3, 5],
                'n_estimators': [1, 501, 1001],
                'colsample_bytree': [0.5, 0.75, 1.0],
                'subsample': [0.5, 0.75, 1.0],
            }
    cv : int, optional
        Number of CV folds for grid search (default: 10).
    n_jobs : int, optional
        Number of parallel jobs for grid search (default: -3).
    n_samples_grid_search : int, optional
        If provided, subsample the CV data to this many rows for faster
        grid search.
    feature_names : list of str, optional
        Feature names to pass to XGBRegressor.

    Returns
    -------
    xgb.XGBRegressor
        An XGBRegressor (configured with best params if grid search was
        performed, default otherwise).
    dict or None
        Best parameters from grid search, or None if no search was run.
    """
    if param_grid is None:
        param_grid = DEFAULT_XGB_PARAM_GRID

    if X_cv is not None and y_cv is not None:
        base_model = xgb.XGBRegressor(feature_names=feature_names)
        grid_search = GridSearchCV(
            base_model, param_grid,
            scoring='r2', cv=cv, verbose=3, n_jobs=n_jobs,
        )

        if n_samples_grid_search is not None:
            X_search = X_cv[:n_samples_grid_search]
            y_search = y_cv[:n_samples_grid_search]
        else:
            X_search = X_cv
            y_search = y_cv

        grid_search.fit(X_search, y_search)
        best_params = grid_search.best_params_
        return xgb.XGBRegressor(**best_params), best_params

    return xgb.XGBRegressor(feature_names=feature_names), None


# ===========================================================================
# PyTorch DNN architectures
# ===========================================================================

def build_fusionage_dnn(input_size,
                        hidden_size=DEFAULT_HIDDEN_SIZE,
                        dropout=DEFAULT_DROPOUT):
    """Build the standard FusionAge DNN architecture.

    Architecture (used for modality-specific and multimodal clocks):
        input -> hidden -> hidden -> hidden -> 32 -> 16 -> 8 -> 1
    Each hidden block consists of Linear -> ReLU -> BatchNorm1d -> Dropout.
    The final reduction layers use Linear -> ReLU -> Dropout.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int, optional
        Width of the three hidden layers (default: 256).
    dropout : float, optional
        Dropout probability (default: 0.05).

    Returns
    -------
    nn.Sequential
        The DNN model.
    """
    return nn.Sequential(
        # Hidden block 1
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),

        # Hidden block 2
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),

        # Hidden block 3
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),

        # Reduction layers
        nn.Linear(hidden_size, 32),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        # Output
        nn.Linear(8, 1),
    )


def build_fusionage_dnn_deep(input_size,
                             hidden_size=DEFAULT_HIDDEN_SIZE,
                             dropout=DEFAULT_DROPOUT):
    """Build the extended FusionAge DNN architecture.

    Architecture (used in hyperparameter tuning experiments):
        input -> hidden -> hidden -> hidden -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1
    Each hidden block consists of Linear -> ReLU -> BatchNorm1d -> Dropout.
    The final reduction layers use Linear -> ReLU -> Dropout.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int, optional
        Width of the three hidden layers (default: 256).
    dropout : float, optional
        Dropout probability (default: 0.05).

    Returns
    -------
    nn.Sequential
        The DNN model.
    """
    return nn.Sequential(
        # Hidden block 1
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),

        # Hidden block 2
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),

        # Hidden block 3
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),

        # Reduction layers
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Dropout(p=dropout),

        # Output
        nn.Linear(8, 1),
    )


# ===========================================================================
# Data preparation (DNN)
# ===========================================================================

def prepare_tensors(X_train, y_train, X_test, y_test):
    """Convert numpy arrays or DataFrames to PyTorch tensors.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector (chronological age).
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        Test target vector (chronological age).

    Returns
    -------
    tuple of torch.Tensor
        (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    """
    X_train_t = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_t = torch.tensor(np.array(y_test), dtype=torch.float32).reshape(-1, 1)
    return X_train_t, y_train_t, X_test_t, y_test_t


# ===========================================================================
# Training (DNN)
# ===========================================================================

def train_dnn(model,
              X_train,
              y_train,
              X_val,
              y_val,
              n_epochs=DEFAULT_N_EPOCHS,
              batch_size=DEFAULT_BATCH_SIZE,
              learning_rate=DEFAULT_LEARNING_RATE,
              weight_decay=DEFAULT_WEIGHT_DECAY,
              verbose=True):
    """Train a FusionAge DNN model with mini-batch SGD and early stopping.

    Uses MSE loss and SGD optimizer. The best model weights (by validation
    MSE) are restored after training completes.

    Parameters
    ----------
    model : nn.Module
        The DNN model to train (e.g., from build_fusionage_dnn).
    X_train : torch.Tensor
        Training features tensor.
    y_train : torch.Tensor
        Training targets tensor, shape (n_samples, 1).
    X_val : torch.Tensor
        Validation features tensor (used for early stopping).
    y_val : torch.Tensor
        Validation targets tensor, shape (n_samples, 1).
    n_epochs : int, optional
        Maximum number of training epochs (default: 300).
    batch_size : int, optional
        Mini-batch size (default: 1000).
    learning_rate : float, optional
        Learning rate for SGD (default: 0.0005).
    weight_decay : float, optional
        L2 regularization strength (default: 0.05).
    verbose : bool, optional
        Whether to print epoch-level progress (default: True).

    Returns
    -------
    model : nn.Module
        The trained model with best weights restored.
    history : list of float
        Validation MSE at each epoch.
    best_mse : float
        Best validation MSE achieved during training.
    """
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_mse = np.inf
    best_weights = None
    history = []

    batch_start = torch.arange(0, len(X_train), batch_size)

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0,
                       disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.set_postfix(mse=float(loss))

        # Evaluate on validation set at end of each epoch
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            mse = float(loss_fn(y_pred_val, y_val))
        history.append(mse)

        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"epoch: {epoch}; val_mse: {mse:.4f}")

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_mse


# ===========================================================================
# Prediction
# ===========================================================================

def predict(model, X):
    """Generate predictions from a trained FusionAge model.

    Automatically detects whether the model is a PyTorch nn.Module or a
    scikit-learn / XGBoost estimator and calls the appropriate prediction
    method.

    Parameters
    ----------
    model : nn.Module or sklearn estimator or xgb.XGBRegressor
        A trained model.
    X : torch.Tensor, np.ndarray, or pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Predicted biological age, shape (n_samples,).
    """
    if isinstance(model, nn.Module):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(X)
        return y_pred.numpy().ravel()
    else:
        return np.array(model.predict(X)).ravel()
