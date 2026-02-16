"""
Cross-validation routines for FusionAge aging clock training.

Provides 10-fold cross-validation wrappers for each regression algorithm
used in the FusionAge framework: scikit-learn linear models, XGBoost, DNN,
and the PhenoAge baseline clock.  Each function returns the best model from
cross-validation along with summary statistics (mean R, best R, 95% margin
of error).
"""

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold

from FusionAge.utils import compute_phenoage, r_score


def cross_validation_regression(X_train,
                     y_train,
                     regressor                 
                     ):
    """Perform 10-fold cross-validation for a scikit-learn regression model.

    Trains and evaluates the regressor across 10 folds, selecting the fold
    with the highest Pearson correlation coefficient (R).  For linear models
    (LinearRegression, LassoCV, ElasticNetCV), model coefficients and
    intercepts are saved from each fold.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : np.ndarray
        Training target vector (chronological age).
    regressor : sklearn estimator
        A scikit-learn compatible regressor (e.g., LinearRegression, LassoCV,
        ElasticNetCV).

    Returns
    -------
    model_best : sklearn estimator
        The fitted model from the fold with the highest R.
    intercept_best : float or np.nan
        Intercept of the best model (np.nan for non-linear models).
    coefs_best : np.ndarray or np.nan
        Coefficients of the best model (np.nan for non-linear models).
    mean_score_r_train_cv : float
        Mean Pearson R across all folds.
    best_r : float
        Best Pearson R across all folds.
    margin_of_error : float
        95% confidence interval margin of error for mean R.
    """
    num_folds = 10
    kf = KFold(n_splits=num_folds)
    scores_r2 = []
    models = []
    weights = []  # To store model weights
    intercepts = []


    cnt_fold = 0
    for train_index, test_index in kf.split(X_train):
        cnt_fold += 1
        print(f"CV Fold: {cnt_fold}")
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Clone the model to ensure each fold starts with a fresh model
        cloned_regressor = clone(regressor)

        # Train the model
        cloned_regressor.fit(X_train_fold, y_train_fold)

        # Evaluate the model
        score_r2 = cloned_regressor.score(X_test_fold, y_test_fold)
        scores_r2.append(score_r2)

        # Save the model name from this fold
        models.append(cloned_regressor)
        
        # save the model weights from this fold
        if type(cloned_regressor) in [sklearn.linear_model._base.LinearRegression,
                                      sklearn.linear_model._coordinate_descent.LassoCV,
                                      sklearn.linear_model._coordinate_descent.ElasticNetCV]:
            weights.append(cloned_regressor.coef_)
            intercepts.append(cloned_regressor.intercept_)
        else:
            weights.append(np.nan)
            intercepts.append(np.nan)

    scores_r = [np.sqrt(x) for x in scores_r2]
    index_cv_best_r = np.nanargmax(scores_r)
    best_r = scores_r[index_cv_best_r]
    model_best = models[index_cv_best_r]
    intercept_best = intercepts[index_cv_best_r]
    coefs_best = weights[index_cv_best_r]

    mean_score_r_train_cv = np.nanmean(scores_r)
    std_dev = np.nanstd(scores_r)

    # Compute the margin of error
    margin_of_error = 1.96 * (std_dev / np.sqrt(len(scores_r)))

    # Compute the confidence interval
    confidence_interval = (mean_score_r_train_cv - margin_of_error, mean_score_r_train_cv + margin_of_error)

    # Print the results
    print('Mean R:', mean_score_r_train_cv)
    print('Best R:', best_r)
    print('Confidence Interval (95%):', confidence_interval)
    print('Stdev: ', std_dev)
    print('Stderr: ', std_dev / np.sqrt(len(scores_r)))
    print('Margin of error: ', margin_of_error)

    return model_best, intercept_best, coefs_best, mean_score_r_train_cv, best_r, margin_of_error


def cross_validation_xgb(X_train,
                     y_train,
                     regressor                 
                     ):
    """Perform 10-fold cross-validation for an XGBoost regressor.

    Trains and evaluates the XGBRegressor across 10 folds using Pearson R
    as the evaluation metric.  Selects and returns the model from the fold
    with the highest R.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : np.ndarray
        Training target vector (chronological age).
    regressor : xgb.XGBRegressor
        An XGBRegressor instance (typically with best hyperparameters from
        a prior GridSearchCV).

    Returns
    -------
    model_best : xgb.XGBRegressor
        The fitted model from the fold with the highest R.
    mean_score_r_train_cv : float
        Mean Pearson R across all folds.
    best_r : float
        Best Pearson R across all folds.
    margin_of_error : float
        95% confidence interval margin of error for mean R.
    """
    num_folds = 10
    kf = KFold(n_splits=num_folds)
    scores_r = []
    models = []

    cnt_fold = 0
    for train_index, test_index in kf.split(X_train):
        cnt_fold += 1
        print(f"CV Fold: {cnt_fold}")
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Instantiate new XGBoost Regressor model
        cloned_regressor = regressor

        # Train the model
        cloned_regressor.fit(X_train_fold, y_train_fold)

        # Evaluate the model
        y_pred_fold = cloned_regressor.predict(X_test_fold)
        score_r = r_score(y_test_fold, y_pred_fold)

        scores_r.append(score_r)

        # Save the model name from this fold
        models.append(cloned_regressor)


    index_cv_best_r = np.argmax(scores_r)
    best_r = scores_r[index_cv_best_r]
    model_best = models[index_cv_best_r]


    mean_score_r_train_cv = np.nanmean(scores_r)
    std_dev = np.nanstd(scores_r)

    # Compute the margin of error
    margin_of_error = 1.96 * (std_dev / np.sqrt(len(scores_r)))

    # Compute the confidence interval
    confidence_interval = (mean_score_r_train_cv - margin_of_error, mean_score_r_train_cv + margin_of_error)

    print('Mean R:', mean_score_r_train_cv)
    print('Best R:', best_r)
    print('Confidence Interval (95%):', confidence_interval)
    print('Stdev: ', std_dev)
    print('Stderr: ', std_dev / np.sqrt(len(scores_r)))
    print('Margin of error: ', margin_of_error)

    return model_best, mean_score_r_train_cv, best_r, margin_of_error


def cross_validation_dnn(X_train,
                     y_train,
                     regressor                 
                     ):
    """Perform 10-fold cross-validation for a PyTorch DNN regressor.

    Placeholder for DNN cross-validation.  DNN training is performed via
    ``FusionAge.model.train_dnn``, which handles mini-batch SGD with early
    stopping on a validation set.  See the model training notebooks for the
    full DNN cross-validation workflow.

    Parameters
    ----------
    X_train : pd.DataFrame or torch.Tensor
        Training feature matrix.
    y_train : np.ndarray or torch.Tensor
        Training target vector (chronological age).
    regressor : nn.Module
        A PyTorch DNN model (e.g., from ``build_fusionage_dnn``).

    Returns
    -------
    None
        Not yet implemented; DNN CV is handled in notebooks.
    """
    return


def cross_validation_phenoage(X_train,
                     y_train):
    """Perform 10-fold cross-validation for the PhenoAge baseline clock.

    Evaluates the hard-coded PhenoAge model (Levine et al. 2018) across
    10 folds.  PhenoAge is not trained — its coefficients are fixed — so
    this function only computes Pearson R on each fold's test set using
    the pre-defined PhenoAge formula.  Infinite or NaN predictions are
    filtered out before scoring.

    The PhenoAge formula requires the following columns in ``X_train``:
    CHEM-BLOOD_Albumin, CHEM-BLOOD_Creatinine, CHEM-BLOOD_Glucose,
    CHEM-BLOOD_C-reactive protein, CHEM-BLOOD_Lymphocyte percentage,
    CHEM-BLOOD_Mean corpuscular volume,
    CHEM-BLOOD_Red blood cell (erythrocyte) distribution width,
    CHEM-BLOOD_Alkaline phosphatase,
    CHEM-BLOOD_White blood cell (leukocyte) count, and age.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix containing PhenoAge biomarker columns and age.
    y_train : np.ndarray
        Target vector (chronological age).

    Returns
    -------
    str
        The string ``"PhenoAge"`` (model identifier).
    mean_score_r_train_cv : float
        Mean Pearson R across all folds.
    best_r : float
        Best Pearson R across all folds.
    margin_of_error : float
        95% confidence interval margin of error for mean R.
    """
    num_folds = 10
    kf = KFold(n_splits=num_folds)
    scores_r = []

    cnt_fold = 0
    for train_index, test_index in kf.split(X_train):
        cnt_fold += 1
        print(f"CV Fold: {cnt_fold}")
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = np.array(y_train).ravel()[train_index], np.array(y_train).ravel()[test_index]

        # Evaluate the model: compute PhenoAge (hard-coded model) on test set
        l_phenoage = []
        for idx in range(len(X_test_fold)):
            this_phenoage = compute_phenoage(X_test_fold.iloc[idx]['CHEM-BLOOD_Albumin'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_Creatinine'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_Glucose'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_C-reactive protein'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_Lymphocyte percentage'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_Mean corpuscular volume'],
                                             X_test_fold.iloc[idx][
                                                 'CHEM-BLOOD_Red blood cell (erythrocyte) distribution width'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_Alkaline phosphatase'],
                                             X_test_fold.iloc[idx]['CHEM-BLOOD_White blood cell (leukocyte) count'],
                                             X_test_fold.iloc[idx]['age']
                                             )
            l_phenoage.append(this_phenoage)


        y_pred_fold = np.array(l_phenoage).ravel()

        # filter out instances where PhenoAge predicts Infinite age
        valid_indices = ~np.isnan(y_test_fold) & ~np.isnan(y_pred_fold) & ~np.isinf(y_test_fold) & ~np.isinf(
            y_pred_fold)
        y_test_fold_clean = y_test_fold[valid_indices]
        y_pred_fold_clean = y_pred_fold[valid_indices]

        score_r = r_score(y_test_fold_clean, y_pred_fold_clean)

        scores_r.append(score_r)

    index_cv_best_r = np.argmax(scores_r)
    best_r = scores_r[index_cv_best_r]


    mean_score_r_train_cv = np.nanmean(scores_r)
    std_dev = np.nanstd(scores_r)

    # Compute the margin of error
    margin_of_error = 1.96 * (std_dev / np.sqrt(len(scores_r)))

    # Compute the confidence interval
    confidence_interval = (mean_score_r_train_cv - margin_of_error, mean_score_r_train_cv + margin_of_error)

    print('Mean R:', mean_score_r_train_cv)
    print('Best R:', best_r)
    print('Confidence Interval (95%):', confidence_interval)
    print('Stdev: ', std_dev)
    print('Stderr: ', std_dev / np.sqrt(len(scores_r)))
    print('Margin of error: ', margin_of_error)

    return "PhenoAge", mean_score_r_train_cv, best_r, margin_of_error
