"""
Performance evaluation and biological age acceleration statistics for FusionAge.

Provides functions for:
    - Computing biological age acceleration (BAA) with bias correction via
      linear regression normalization (adapted from McIntyre et al. 2021,
      MoveAge: https://www.frontiersin.org/articles/10.3389/fragi.2021.708680)
    - Generating comprehensive performance evaluation DataFrames with
      Pearson R, MAE, MSE, RMSE for both raw and corrected predictions
    - Computing age acceleration in 5-year chronological age windows
"""

import numpy as np
import pandas as pd
from FusionAge.utils import r_score, r2_score
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Bias correction: normalized BA - adapted from McIntyre et al. 2021 (MoveAge)
"""
* paper: https://www.frontiersin.org/articles/10.3389/fragi.2021.708680/full
* method:
1.  A random forest model was generated on centered and scaled data from the training dataset using the randomForest package in R (Breiman and Cutler, 2018). Model parameters were assessed using the randomForest and randomForestExplainer packages in R (Breiman and Cutler, 2018; Paluszynska et al., 2020).
    1a. The model was validated using the 2005–2006 validation dataset. 
    
2.  Finally, to ensure a linear relation between predicted and chronological age, an individual’s predicted age was normalized by dividing by the median predicted ages of individuals of similar chronological ages and multiplying again by the individual’s actual chronological age. 

3.  Age acceleration and deceleration was then evaluated by subtracting an individual’s chronological age from their normalized biological age prediction. See supplementary materials for extended methods.
"""


def create_age_accleration_and_performance_eval_df(df_eid_TRAIN,
                                                   df_eid_TEST,
                                                   y_train,
                                                   y_pred_train,
                                                   y_test,
                                                   y_pred,
                                                   X_train,
                                                   X_cv,
                                                   X_test,
                                                   mean_score_r_train_cv,
                                                   margin_of_error,
                                                   s_feature_set,
                                                   this_model
                                                   ):
    """Compute biological age acceleration and compile a performance evaluation DataFrame.

    For both train and test sets, this function:
    1. Fits a linear regression of BA ~ CA on the training set to obtain the
       expected median BA for each CA (bias correction).
    2. Computes biological age acceleration (BAA = BA_uncorrected - BA_median_per_ca).
    3. Derives bias-corrected BA (BA_corrected = CA + BAA), normalized BAA,
       and z-scored BAA.
    4. Computes performance metrics (Pearson R, MAE, MSE, RMSE) for both
       raw and corrected predictions on train and test sets.

    The test set BAA is computed using the slope and intercept from the
    training set regression, ensuring no data leakage.

    Parameters
    ----------
    df_eid_TRAIN : pd.DataFrame
        Training set DataFrame containing an 'eid' column.
    df_eid_TEST : pd.DataFrame
        Test set DataFrame containing an 'eid' column.
    y_train : np.ndarray
        True chronological ages for the training set.
    y_pred_train : np.ndarray
        Predicted biological ages for the training set.
    y_test : np.ndarray
        True chronological ages for the test set.
    y_pred : np.ndarray
        Predicted biological ages for the test set.
    X_train : pd.DataFrame
        Training feature matrix (used for sample count).
    X_cv : pd.DataFrame
        Cross-validation feature matrix (used for sample count).
    X_test : pd.DataFrame
        Test feature matrix (used for sample count and feature count).
    mean_score_r_train_cv : float
        Mean Pearson R from cross-validation on the training set.
    margin_of_error : float
        95% CI margin of error from cross-validation on the training set.
    s_feature_set : str
        Label for the feature set used.
    this_model : object
        The trained model object.

    Returns
    -------
    df_eid_ca_ba_median_predictions_TRAIN : pd.DataFrame
        Training set DataFrame with columns: eid, CA, BA_uncorrected,
        BA_median_per_ca, BAA, BA_corrected, BAA_norm, BAA_zscore.
    df_eid_ca_ba_median_predictions_TEST : pd.DataFrame
        Test set DataFrame with the same columns as above.
    df_performance : pd.DataFrame
        Transposed performance summary with metrics: R, MAE, MSE, RMSE
        (raw and corrected), sample counts, and feature set metadata.
    """
    ## predictions for each eid
    df_eid_ca_ba_median_predictions_TRAIN = pd.DataFrame([])
    df_eid_ca_ba_median_predictions_TRAIN['eid'] = df_eid_TRAIN['eid']
    df_eid_ca_ba_median_predictions_TRAIN['CA'] = y_train
    df_eid_ca_ba_median_predictions_TRAIN['BA_uncorrected'] = y_pred_train
    age_accel_results_TRAIN = compute_age_acceleration(df_eid_ca_ba_median_predictions_TRAIN['CA'],
                                                       df_eid_ca_ba_median_predictions_TRAIN['BA_uncorrected'])
    age_accel_results_TRAIN_slope = age_accel_results_TRAIN[3]
    age_accel_results_TRAIN_intercept = age_accel_results_TRAIN[2]
    print(age_accel_results_TRAIN_intercept, age_accel_results_TRAIN_slope)
    # compute expected median biological age BA for this particular chronological age CA
    df_eid_ca_ba_median_predictions_TRAIN['BA_median_per_ca'] = age_accel_results_TRAIN[0]
    df_eid_ca_ba_median_predictions_TRAIN['BAA'] = df_eid_ca_ba_median_predictions_TRAIN['BA_uncorrected'] - \
                                                   df_eid_ca_ba_median_predictions_TRAIN['BA_median_per_ca']
    # the corrected BA is a BA that is shifted towards the mean
    df_eid_ca_ba_median_predictions_TRAIN['BA_corrected'] = df_eid_ca_ba_median_predictions_TRAIN['CA'] + \
                                                            df_eid_ca_ba_median_predictions_TRAIN['BAA']
    df_eid_ca_ba_median_predictions_TRAIN['BAA_norm'] = df_eid_ca_ba_median_predictions_TRAIN['BAA'] / \
                                                        df_eid_ca_ba_median_predictions_TRAIN['CA']
    df_eid_ca_ba_median_predictions_TRAIN['BAA_zscore'] = stats.zscore(df_eid_ca_ba_median_predictions_TRAIN['BAA'])

    df_eid_ca_ba_median_predictions_TEST = pd.DataFrame([])
    df_eid_ca_ba_median_predictions_TEST['eid'] = df_eid_TEST['eid']
    df_eid_ca_ba_median_predictions_TEST['CA'] = y_test
    df_eid_ca_ba_median_predictions_TEST['BA_uncorrected'] = y_pred

    # for TEST set, compute the BAA (Biological Age Acceleration / age gap) using the slope and intercept from
    # compute_age_accleration() on the training set
    df_eid_ca_ba_median_predictions_TEST[
        'BA_median_per_ca'] = age_accel_results_TRAIN_intercept + age_accel_results_TRAIN_slope * \
                              df_eid_ca_ba_median_predictions_TEST['CA']
    # comptue age acceleration BAA (age gap) by finding difference between model-predicted biological age (BA_uncorrected)
    # and the expected median biological age (column 'BA_median_per_ca')
    df_eid_ca_ba_median_predictions_TEST['BAA'] = df_eid_ca_ba_median_predictions_TEST['BA_uncorrected'] - \
                                                  df_eid_ca_ba_median_predictions_TEST['BA_median_per_ca']
    # the corrected BA is a BA that is shifted towards the mean
    df_eid_ca_ba_median_predictions_TEST['BA_corrected'] = df_eid_ca_ba_median_predictions_TEST['CA'] + \
                                                           df_eid_ca_ba_median_predictions_TEST['BAA']
    df_eid_ca_ba_median_predictions_TEST['BAA_norm'] = df_eid_ca_ba_median_predictions_TEST['BAA'] / \
                                                       df_eid_ca_ba_median_predictions_TEST['CA']
    df_eid_ca_ba_median_predictions_TEST['BAA_zscore'] = stats.zscore(df_eid_ca_ba_median_predictions_TEST['BAA'])

    y_pred_train_corrected = np.array(df_eid_ca_ba_median_predictions_TRAIN['BA_corrected'])
    y_pred_test_corrected = np.array(df_eid_ca_ba_median_predictions_TEST['BA_corrected'])

    r_train = r_score(y_train, y_pred_train)
    r_test = r_score(y_test, y_pred)
    r_train_corrected = r_score(y_train, y_pred_train_corrected)
    r2_train_corrected = r2_score(y_train, y_pred_train_corrected)
    r_test_corrected = r_score(y_test, y_pred_test_corrected)
    r2_test_corrected = r2_score(y_test, y_pred_test_corrected)

    print('r_train: ', r_train,
          '\nr_test: ', r_test,
          '\nr_train_corrected: ', r_train_corrected,
          '\nr2_train_corrected: ', r2_train_corrected,
          '\nr_test_corrected: ', r_test_corrected,
          '\nr2_test_corrected: ', r2_test_corrected
          )

    # Calcualte mean absolute error
    mae_train = mean_absolute_error(y_train, y_pred_train)
    # Calculate mean squared error
    mse_train = mean_squared_error(y_train, y_pred_train)
    # Calculate root mean squared error
    rmse_train = np.sqrt(mse_train)

    # Calcualte mean absolute error
    mae_train_corrected = mean_absolute_error(y_train, y_pred_train_corrected)
    # Calculate mean squared error
    mse_train_corrected = mean_squared_error(y_train, y_pred_train_corrected)
    # Calculate root mean squared error
    rmse_train_corrected = np.sqrt(mse_train_corrected)

    # Calcualte mean absolute error
    mae_test = mean_absolute_error(y_test, y_pred)
    # Calculate mean squared error
    mse_test = mean_squared_error(y_test, y_pred)
    # Calculate root mean squared error
    rmse_test = np.sqrt(mse_test)

    # Calcualte mean absolute error
    mae_test_corrected = mean_absolute_error(y_test, y_pred_test_corrected)
    # Calculate mean squared error
    mse_test_corrected = mean_squared_error(y_test, y_pred_test_corrected)
    # Calculate root mean squared error
    rmse_test_corrected = np.sqrt(mse_test_corrected)

    # Print the MSE, RMSE, MAE: train
    print("MSE_train:", mse_train)
    print("MAE_train:", mae_train)
    print("RMSE_train:", rmse_train)
    print("MAE_train_corrected:", mae_train_corrected)
    print("MSE_train_corrected:", mse_train_corrected)
    print("RMSE_train_corrected:", rmse_train_corrected)

    # Print the MSE, RMSE, MAE: test
    print("MSE_test:", mse_test)
    print("MAE_test:", mae_test)
    print("RMSE_test:", rmse_test)
    print("MAE_test_corrected:", mae_test_corrected)
    print("MSE_test_corrected:", mse_test_corrected)
    print("RMSE_test_corrected:", rmse_test_corrected)

    # performance evaluation
    n_pts = len(X_train) + len(X_cv) + len(X_test)

    df_performance = pd.DataFrame([])
    df_performance["r_train"] = [r_train]  # R on training set with final feature set
    df_performance["r_test"] = [r_test]
    df_performance["r_train_corrected"] = [r_train_corrected]
    df_performance["r_test_corrected"] = [r_test_corrected]
    df_performance["n_pts"] = n_pts
    df_performance["n_train"] = [len(X_train)]
    df_performance["n_cv"] = [len(X_cv)]
    df_performance["n_test"] = [len(X_test)]
    df_performance["n_features"] = [X_test.shape[1]]
    df_performance[
        "r_train_cv"] = mean_score_r_train_cv  # cross-validation on training set with final feature set, to use for confidence intervals
    df_performance[
        "r_train_cv_margin_of_error"] = margin_of_error  # margin of error (1.96 std err), to use for confidence interval error bars

    df_performance["MAE_train"] = [mae_train]
    df_performance["MSE_train"] = [mse_train]
    df_performance["RMSE_train"] = [rmse_train]
    df_performance["MAE_train_corrected"] = [mae_train_corrected]
    df_performance["MSE_train_corrected"] = [mse_train_corrected]
    df_performance["RMSE_train_corrected"] = [rmse_train_corrected]
    df_performance["MAE_test"] = [mae_test]
    df_performance["MSE_test"] = [mse_test]
    df_performance["RMSE_test"] = [rmse_test]
    df_performance["MAE_test_corrected"] = [mae_test_corrected]
    df_performance["MSE_test_corrected"] = [mse_test_corrected]
    df_performance["RMSE_test_corrected"] = [rmse_test_corrected]

    df_performance["feature_set"] = [s_feature_set]
    df_performance["model"] = [this_model]
    df_performance = df_performance.T

    return (df_eid_ca_ba_median_predictions_TRAIN,
            df_eid_ca_ba_median_predictions_TEST,
            df_performance)


def compute_age_acceleration(l_chronological_age,
                             l_biological_age):
    """
    For a given list of chronological ages and biological ages, compute the age acceleration and store in a separate list
    # Example usage:
    chronological_ages = [25, 30, 35, 40, 45]
    biological_ages = [28, 31, 37, 43, 50]
    age_accelerations, intercept, slope = compute_age_acceleration(chronological_ages, biological_ages)

    print("Age Accelerations:", age_accelerations)
    print("Intercept:", intercept)
    print("Slope:", slope)

    :param l_chronological_age: List of integers or floats representing chronological ages
    :param l_biological_age: List of integers or floats representing biological ages
    :return: Tuple containing:
             - List of age accelerations
             - Intercept of regression BA~CA
             - Beta (slope) of regression BA~CA



    """

    # Convert lists to numpy arrays for easier mathematical operations
    ca = np.array(l_chronological_age)
    ba = np.array(l_biological_age)

    # Compute regression line
    slope, intercept, r_value, p_value, std_err = linregress(ca, ba)

    # Predict biological ages using the regression line
    predicted_ba = intercept + slope * ca

    # Compute the difference between the biological age and the regression line to get age acceleration
    l_age_acceleration = ba - predicted_ba

    return predicted_ba, list(l_age_acceleration), intercept, slope


def compute_age_acceleration_chunk_5yr(l_chronological_age, l_biological_age):
    """Compute age acceleration within 5-year chronological age windows.

    Instead of fitting a single global regression of BA ~ CA, this function
    partitions individuals into 5-year age bins and fits a separate linear
    regression within each bin.  Age acceleration for each individual is
    computed relative to their age-bin-specific regression line, which
    accounts for potential non-linearity in the BA ~ CA relationship.

    Parameters
    ----------
    l_chronological_age : array-like
        Chronological ages.
    l_biological_age : array-like
        Biological ages (model predictions).

    Returns
    -------
    l_predicted_biological_age : list of float
        Expected (regression-predicted) biological ages for each individual.
    l_age_acceleration : list of float
        Age acceleration values (BA - predicted BA) for each individual.
    intercepts : dict
        Mapping of age-bin start -> regression intercept.
    slopes : dict
        Mapping of age-bin start -> regression slope.
    """
    l_chronological_age = np.array(l_chronological_age)
    l_biological_age = np.array(l_biological_age)

    # Initialize the list for age accelerations
    l_age_acceleration = []
    l_predicted_biological_age = []

    # Define the range for age groups based on 5-year windows
    age_min = l_chronological_age.min()
    age_max = l_chronological_age.max()
    age_ranges = range(int(age_min), int(age_max) + 5, 5)

    # Dictionary to hold intercepts and slopes for each group
    intercepts = {}
    slopes = {}

    # Calculate regression for each age group
    for start_age in age_ranges:
        mask = (l_chronological_age >= start_age) & (l_chronological_age < start_age + 5)
        if np.sum(mask) > 1:  # At least two points to fit a line
            slope, intercept, _, _, _ = linregress(l_chronological_age[mask], l_biological_age[mask])
            intercepts[start_age] = intercept
            slopes[start_age] = slope

            # Calculate acceleration for this age group
            predicted_biological_age = intercept + slope * l_chronological_age[mask]
            acceleration = l_biological_age[mask] - predicted_biological_age
            l_predicted_biological_age.extend(predicted_biological_age.tolist())
            l_age_acceleration.extend(acceleration.tolist())
        else:
            # Not enough data to fit a regression, possibly handle differently
            l_age_acceleration.extend([0] * np.sum(mask))  # Default to no acceleration

    return l_predicted_biological_age,l_age_acceleration, intercepts, slopes