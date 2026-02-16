"""
Utility functions for FusionAge aging clock evaluation and interpretability.

Includes:
    - PhenoAge baseline clock computation (Levine et al. 2018)
    - Residual plot generation (biological age vs. chronological age)
    - SHAP-based interpretability for linear, XGBoost, and DNN models
    - Pearson correlation scoring utilities
    - Shapley feature ranking
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from scipy.stats import gaussian_kde
import shap


def compute_phenoage(albumin, creatinine, glucose, crp, lymph_percent, mcv, rdw, alp, wbc, age):
    """Compute PhenoAge (Levine et al. 2018) for a single individual.

    Implements the PhenoAge formula using nine blood chemistry biomarkers
    and chronological age.  Used as a baseline aging clock for comparison
    against FusionAge-trained clocks.

    Parameters
    ----------
    albumin : float
        Serum albumin (g/L).
    creatinine : float
        Serum creatinine (umol/L).
    glucose : float
        Fasting glucose (mmol/L).
    crp : float
        C-reactive protein (mg/L).
    lymph_percent : float
        Lymphocyte percentage (%).
    mcv : float
        Mean corpuscular volume (fL).
    rdw : float
        Red blood cell distribution width (%).
    alp : float
        Alkaline phosphatase (U/L).
    wbc : float
        White blood cell count (10^9/L).
    age : float
        Chronological age (years).

    Returns
    -------
    float
        Estimated PhenoAge (years).

    Reference
    ---------
    Levine, M. E. et al. An epigenetic biomarker of aging for lifespan and
    healthspan. Aging 10, 573-591 (2018).
    """
    sum_coefftimesvalue = (-0.0336) * albumin + \
                          0.0095 * creatinine + \
                          0.1953 * glucose + \
                          0.0954 * crp + \
                          (-0.0120) * lymph_percent + \
                          0.0268 * mcv + \
                          0.3306 * rdw + \
                          0.0019 * alp + \
                          0.0554 * wbc + \
                          0.0804 * age
    LC = -19.9067 + sum_coefftimesvalue
    gamma = 0.0076927
    mortality_score = 1 - np.exp(-np.exp(LC) * (np.exp(120 * gamma) - 1) / gamma)
    pheno_age = 141.50225 + np.log(-0.00553 * np.log(1 - mortality_score)) / 0.09165

    return pheno_age


def make_residual_plot(y_train, y_test, y_pred, y_pred_train,
                       result_dir_this_model, model, s_feature_set,
                       graph_chronological_age_min=30,
                       graph_chronological_age_max=80):
    """Generate a 2x3 panel of residual plots for biological vs. chronological age.

    Creates density scatter plots (top row) and residual histograms (bottom
    row) for three splits: train+test combined, train only, and test only.
    Points are colored by local density using a Gaussian KDE.  Figures are
    saved as both SVG and PNG to the specified output directory.

    Parameters
    ----------
    y_train : np.ndarray
        True chronological ages for the training set.
    y_test : np.ndarray
        True chronological ages for the test set.
    y_pred : np.ndarray
        Predicted biological ages for the test set.
    y_pred_train : np.ndarray
        Predicted biological ages for the training set.
    result_dir_this_model : str
        Directory path where figures will be saved.
    model : object
        The trained model (used for naming the output file).
    s_feature_set : str
        String label for the feature set (used for naming the output file).
    graph_chronological_age_min : int, optional
        Minimum age for plot axes (default: 30).
    graph_chronological_age_max : int, optional
        Maximum age for plot axes (default: 80).
    """
    fig, ax = plt.subplots(figsize=(18, 9), nrows=2, ncols=3)

    ## TRAIN + TEST ######################################################################################
    s_title = 'model scored on \nTRAIN+TEST'
    x = np.concatenate([y_test.ravel(), y_train.ravel()])[:10000]
    y = np.concatenate([y_pred.ravel(), y_pred_train.ravel()])[:10000]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # Compute the density of points

    scatter = ax[0][0].scatter(x, y, c=z, s=50, edgecolor='none', cmap='viridis')  # Points colored by density
    ax[0][0].plot([graph_chronological_age_min, graph_chronological_age_max], [graph_chronological_age_min, graph_chronological_age_max], 'k--', lw=1)  # Add the line

    # Set the aspect of the plot to be equal
    ax[0][0].set_aspect('equal')

    # Set ticks every 5 units
    ax[0][0].set_xticks(np.arange(graph_chronological_age_min, graph_chronological_age_max, 5))
    ax[0][0].set_yticks(np.arange(graph_chronological_age_min, graph_chronological_age_max, 5))

    # Optionally, you can also adjust the plot limits if necessary
    ax[0][0].set_xlim(graph_chronological_age_min, graph_chronological_age_max)
    ax[0][0].set_ylim(graph_chronological_age_min, graph_chronological_age_max)

    # title, axes labels
    ax[0][0].set_title(s_title)
    ax[0][0].set_xlabel("Chronological age")
    ax[0][0].set_ylabel("Biological age")

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax[0][0])
    cbar.set_label('Density')

    # histogram
    df_residuals_raw = pd.DataFrame({'CA': x,
                                     'BA': y})
    df_residuals_raw['residual'] = df_residuals_raw['BA'] - df_residuals_raw['CA']
    ax[1][0].hist(df_residuals_raw['residual'], bins=50)
    ax[1][0].set_xlabel("Residual")
    ax[1][0].set_ylabel("Count")

    ## TRAIN ##########################################################################################
    s_title = 'model scored on \nTRAIN'
    x = y_train.ravel()[:10000]
    y = y_pred_train.ravel()[:10000]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # Compute the density of points

    scatter = ax[0][1].scatter(x, y, c=z, s=50, edgecolor='none', cmap='viridis')  # Points colored by density
    ax[0][1].plot([graph_chronological_age_min, graph_chronological_age_max], [graph_chronological_age_min, graph_chronological_age_max], 'k--', lw=1)  # Add the line

    # Set the aspect of the plot to be equal
    ax[0][1].set_aspect('equal')

    # Set ticks every 5 units
    ax[0][1].set_xticks(np.arange(graph_chronological_age_min, graph_chronological_age_max, 5))
    ax[0][1].set_yticks(np.arange(graph_chronological_age_min, graph_chronological_age_max, 5))

    # Optionally, you can also adjust the plot limits if necessary
    ax[0][1].set_xlim(graph_chronological_age_min, graph_chronological_age_max)
    ax[0][1].set_ylim(graph_chronological_age_min, graph_chronological_age_max)

    # title, axes labels
    ax[0][1].set_title(s_title)
    ax[0][1].set_xlabel("Chronological age")
    ax[0][1].set_ylabel("Biological age")

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax[0][1])
    cbar.set_label('Density')

    # histogram
    df_residuals_raw = pd.DataFrame({'CA': x,
                                     'BA': y})
    df_residuals_raw['residual'] = df_residuals_raw['BA'] - df_residuals_raw['CA']
    ax[1][1].hist(df_residuals_raw['residual'], bins=50)
    ax[1][1].set_xlabel("Residual")
    ax[1][1].set_ylabel("Count")

    ## TEST ########################################################################################
    s_title = 'model scored on \nTEST'
    x = y_test.ravel()[:10000]
    y = y_pred.ravel()[:10000]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # Compute the density of points

    scatter = ax[0][2].scatter(x, y, c=z, s=50, edgecolor='none', cmap='viridis')  # Points colored by density
    ax[0][2].plot([graph_chronological_age_min, graph_chronological_age_max], [graph_chronological_age_min, graph_chronological_age_max], 'k--', lw=1)  # Add the line

    # Set the aspect of the plot to be equal
    ax[0][2].set_aspect('equal')

    # Set ticks every 5 units
    ax[0][2].set_xticks(np.arange(graph_chronological_age_min, graph_chronological_age_max, 5))
    ax[0][2].set_yticks(np.arange(graph_chronological_age_min, graph_chronological_age_max, 5))

    # Optionally, you can also adjust the plot limits if necessary
    ax[0][2].set_xlim(graph_chronological_age_min, graph_chronological_age_max)
    ax[0][2].set_ylim(graph_chronological_age_min, graph_chronological_age_max)

    # title, axes labels
    ax[0][2].set_title(s_title)
    ax[0][2].set_xlabel("Chronological age")
    ax[0][2].set_ylabel("Biological age")

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax[0][2])
    cbar.set_label('Density')

    # histogram
    df_residuals_raw = pd.DataFrame({'CA': x,
                                     'BA': y})
    df_residuals_raw['residual'] = df_residuals_raw['BA'] - df_residuals_raw['CA']
    ax[1][2].hist(df_residuals_raw['residual'], bins=50)
    ax[1][2].set_xlabel("Residual")
    ax[1][2].set_ylabel("Count")

    # save the figure
    s_model_name = 'model_imputemissing_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(result_dir_this_model + "/residual_plot_" + s_model_name + '.svg', dpi=600)
    s_model_name = 'model_imputemissing_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(result_dir_this_model + "/residual_plot_" + s_model_name + '.png', dpi=600)



def make_shap_plot_linear(model,
                   X_for_background,
                   X_to_plot,
                   s_feature_set,
                   dir_to_save_plot,
                   n_samples_to_use_for_background=1000):
    """Generate SHAP interpretability plots for a linear model.

    Uses ``shap.LinearExplainer`` to compute SHAP values, then produces a
    summary plot.  Saves the SHAP summary figure (SVG + PNG), a CSV of
    per-individual SHAP values, and the pickled explainer object to the
    specified directory.

    Parameters
    ----------
    model : sklearn linear model
        A fitted linear model (e.g., LinearRegression, LassoCV, ElasticNetCV).
    X_for_background : pd.DataFrame
        Background dataset for the SHAP explainer (typically training data).
    X_to_plot : pd.DataFrame
        Feature matrix to compute and plot SHAP values for (typically test data).
    s_feature_set : str
        Feature set label (used in output filenames).
    dir_to_save_plot : str
        Directory path to save output files.
    n_samples_to_use_for_background : int, optional
        Number of background samples for the explainer (default: 1000).

    Returns
    -------
    shap_values : np.ndarray
        SHAP values matrix, shape (n_samples, n_features).
    explainer : shap.LinearExplainer
        The fitted SHAP explainer object.
    """
    for x in range(300):
        print(x)

    explainer = shap.LinearExplainer(model, X_for_background[:n_samples_to_use_for_background])
    shap_values = explainer.shap_values(X_to_plot)

    # Plot the SHAP values

    fig, ax = plt.subplots(figsize=(10, 20))

    shap.summary_plot(shap_values, X_to_plot, max_display=200)

    # save the figure
    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(dir_to_save_plot + "/SHAP_plot_" + s_model_name + '.svg', dpi=600)
    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(dir_to_save_plot + "/SHAP_plot_" + s_model_name + '.png', dpi=600)

    # save the shap values - TEST DATA
    df_shap_values = pd.DataFrame(shap_values, columns=X_to_plot.columns)
    df_shap_values.to_csv(dir_to_save_plot + "/SHAP_values_" + s_model_name + '.csv', index=False)

    # save the explainer

    # Save the SHAP explainer object to disk
    explainer_path = dir_to_save_plot + '/SHAP_explainer_' + s_model_name + '.pkl'
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)

    return shap_values, explainer



def make_shap_plot_xgb(model,
                   X_to_plot,
                   s_feature_set,
                   dir_to_save_plot):
    """Generate SHAP interpretability plots for an XGBoost model.

    Uses ``shap.Explainer`` to compute SHAP values, then produces:
    1. A SHAP summary (beeswarm) plot of feature importances.
    2. An aggregate contribution score bar chart showing whether each
       feature drives biological age higher or lower on average.

    Also computes and prints the FusionAge aggregate contribution score
    (SHAP value * feature value, averaged across individuals) for the top
    20 features.

    Saves the SHAP plots (SVG + PNG), the aggregate score plot, a CSV
    of per-individual SHAP values, and the pickled explainer object.

    Parameters
    ----------
    model : xgb.XGBRegressor
        A fitted XGBoost model.
    X_to_plot : pd.DataFrame
        Feature matrix to compute and plot SHAP values for (typically test data).
    s_feature_set : str
        Feature set label (used in output filenames).
    dir_to_save_plot : str
        Directory path to save output files.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values matrix, shape (n_samples, n_features).
    explainer : shap.Explainer
        The fitted SHAP explainer object.
    """
    for x in range(300):
        print(x)

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_to_plot)

    # Plot the SHAP values

    fig, ax = plt.subplots(figsize=(10, 20))

    shap.summary_plot(shap_values[:1000], X_to_plot[:1000], max_display=200)

    # save the figure
    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(dir_to_save_plot + "/SHAP_plot_" + s_model_name + '.svg', dpi=600)
    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(dir_to_save_plot + "/SHAP_plot_" + s_model_name + '.png', dpi=600)

    # get a rank list of features as computed by SHAP (using SHAP's criteria)
    df_shap_feature_ranking = shapley_feature_ranking(shap_values, X_to_plot)
    l_shap_ranked_list = np.array(df_shap_feature_ranking['features'])

    # create SCORE for all <patient, feature> combos - store as df
    score_shap_feature_contributions = shap_values * X_to_plot

    sorted_score_shap_feature_contributions = score_shap_feature_contributions[l_shap_ranked_list]

    # compute the mean score for each feature; this gives a pandas Series where each element is a score
    # that represents the relative contribution to biological age
    sorted_mean_score_shap_feature_contributions = np.mean(sorted_score_shap_feature_contributions, axis = 0)


    # print the mean contribution scores for the top 20 features (sorted by our score for feature contributions)
    top_mean_score_shap_feature_contributions = sorted_mean_score_shap_feature_contributions.head(20)
    print(top_mean_score_shap_feature_contributions)

    # visualization
    reversed_series = top_mean_score_shap_feature_contributions[::-1]

    fig, ax = plt.subplots(figsize=(10, 10))  # define figure
    reversed_series.plot(kind='barh', figsize=(14, 7), color='skyblue')
    plt.title('does the feature make you older or younger?', fontsize=16)
    plt.xlabel('<-- Younger; Older -->', fontsize=14)
    plt.ylabel('SHAP Values', fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Annotate each bar with its respective value
    for i, v in enumerate(reversed_series):
        ax.text(v + 0.02, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

    print("TOP FEATURES CONTRIBUTING TO HIGH AGE by LEFT or RIGHT skew of SHAP value given feature value: ")
    print(top_mean_score_shap_feature_contributions[top_mean_score_shap_feature_contributions > 0])

    print("\n\nTOP FEATURES CONTRIBUTING TO LOW AGE by LEFT or RIGHT skew of SHAP value given feature value")
    print(top_mean_score_shap_feature_contributions[top_mean_score_shap_feature_contributions <= 0])

    # save the figure
    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(dir_to_save_plot + "/SHAP_aggregated_score_plot_" + s_model_name + '.svg', dpi=600)
    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set
    fig.savefig(dir_to_save_plot + "/SHAP_aggregated_score_plot_" + s_model_name + '.png', dpi=600)

    # save the shap values - TEST DATA
    df_shap_values = pd.DataFrame(shap_values, columns=X_to_plot.columns)
    df_shap_values.to_csv(dir_to_save_plot + "/SHAP_values_" + s_model_name + '.csv', index=False)

    # save the explainer

    # Save the SHAP explainer object to disk
    explainer_path = dir_to_save_plot + '/SHAP_explainer_' + s_model_name + '.pkl'
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)

    return shap_values, explainer

def make_shap_plot_deep(model,
                        X_for_background,
                        X_to_plot,
                        df_X_to_plot,
                        s_feature_set,
                        dir_to_save_plot,
                        feature_names,
                        n_samples_to_use_for_background=1000,
                        n_samples_to_compute_shap_values_for=1000):
    """Generate SHAP interpretability plots for a PyTorch DNN model.

    Uses ``shap.DeepExplainer`` to compute SHAP values for the DNN, then
    produces:
    1. A SHAP summary (beeswarm) plot of feature importances.
    2. An aggregate contribution score bar chart showing whether each
       feature drives biological age higher or lower on average.

    Also computes and prints the FusionAge aggregate contribution score
    (SHAP value * feature value, averaged across individuals) for the top
    20 features.

    Saves the SHAP plots (SVG + PNG), the aggregate score plot, a CSV
    of per-individual SHAP values, and the pickled explainer object.

    Parameters
    ----------
    model : nn.Module
        A trained PyTorch DNN model.
    X_for_background : torch.Tensor
        Background data tensor for the DeepExplainer (typically training data).
    X_to_plot : torch.Tensor
        Test data tensor to compute SHAP values for.
    df_X_to_plot : pd.DataFrame
        Test data as a DataFrame (same data as X_to_plot, used for feature
        names and aggregate score computation).
    s_feature_set : str
        Feature set label (used in output filenames).
    dir_to_save_plot : str
        Directory path to save output files.
    feature_names : list of str
        Feature names for labeling the SHAP summary plot.
    n_samples_to_use_for_background : int, optional
        Number of background samples for the DeepExplainer (default: 1000).
    n_samples_to_compute_shap_values_for : int or str, optional
        Number of test samples to compute SHAP values for, or ``'all'`` to
        use all samples (default: 1000).

    Returns
    -------
    shap_values : np.ndarray
        SHAP values matrix, shape (n_samples, n_features).
    explainer : shap.DeepExplainer
        The fitted SHAP explainer object.
    """
    background = X_for_background[:n_samples_to_use_for_background].detach()
    if n_samples_to_compute_shap_values_for == 'all':
        print("shap values for all samples")
        n_samples_to_compute_shap_values_for = X_to_plot.shape[
            0]  # reassign to number of individuals for purpose of aggregate SHAP score analysis
        test_data = X_to_plot.detach()  # Data to explain
    else:
        test_data = X_to_plot[:n_samples_to_compute_shap_values_for].detach()  # Data to explain

    print("compute shap values on test_data", test_data.shape)
    # Create the SHAP DeepExplainer
    explainer = shap.DeepExplainer(model,
                                   background)

    shap_values = explainer.shap_values(test_data)

    for x in range(300):
        print(x)

    fig, ax = plt.subplots(figsize=(30, 10))

    # Plot the SHAP values
    shap.summary_plot(shap_values[:1000],  # just use random sample 1000 values for plot
                      test_data[:1000],
                      feature_names=feature_names,
                      max_display=2000)

    ax.tick_params(axis='x', labelsize=2)

    # Increase the width of the bars
    for patch in ax.patches:
        patch.set_width(0.4)  # Adjust the width as desired

    s_model_name = 'model_' + str(model).split('(')[0] + '_' + s_feature_set

    # save the figure
    fig.savefig(dir_to_save_plot + "/SHAP_plot_" + s_model_name + '.svg', dpi=100)
    fig.savefig(dir_to_save_plot + "/SHAP_plot_" + s_model_name + '.png', dpi=100)

    # get a rank list of features as computed by SHAP (using SHAP's criteria)
    df_shap_feature_ranking = shapley_feature_ranking(shap_values,
                                                      df_X_to_plot[:n_samples_to_compute_shap_values_for])
    l_shap_ranked_list = np.array(df_shap_feature_ranking['features'])

    # create SCORE for all <patient, feature> combos - store as df
    score_shap_feature_contributions = shap_values * df_X_to_plot[:n_samples_to_compute_shap_values_for]

    sorted_score_shap_feature_contributions = score_shap_feature_contributions[l_shap_ranked_list]

    # compute the mean score for each feature; this gives a pandas Series where each element is a score
    # that represents the relative contribution to biological age
    sorted_mean_score_shap_feature_contributions = np.mean(sorted_score_shap_feature_contributions, axis=0)

    # print the mean contribution scores for the top 20 features (sorted by our score for feature contributions)
    top_mean_score_shap_feature_contributions = sorted_mean_score_shap_feature_contributions.head(20)
    print(top_mean_score_shap_feature_contributions)

    # visualization
    reversed_series = top_mean_score_shap_feature_contributions[::-1]

    fig, ax = plt.subplots(figsize=(10, 10))  # define figure
    reversed_series.plot(kind='barh', figsize=(14, 7), color='skyblue')
    plt.title('does the feature make you older or younger?', fontsize=16)
    plt.xlabel('<-- Younger; Older -->', fontsize=14)
    plt.ylabel('SHAP Values', fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Annotate each bar with its respective value
    for i, v in enumerate(reversed_series):
        ax.text(v + 0.02, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

    print("TOP FEATURES CONTRIBUTING TO HIGH AGE by LEFT or RIGHT skew of SHAP value given feature value: ")
    print(top_mean_score_shap_feature_contributions[top_mean_score_shap_feature_contributions > 0])

    print("\n\nTOP FEATURES CONTRIBUTING TO LOW AGE by LEFT or RIGHT skew of SHAP value given feature value")
    print(top_mean_score_shap_feature_contributions[top_mean_score_shap_feature_contributions <= 0])

    print(shap_values)

    # save the figure
    fig.savefig(dir_to_save_plot + "/SHAP_aggregated_score_plot_" + s_model_name + '.svg', dpi=600)
    fig.savefig(dir_to_save_plot + "/SHAP_aggregated_score_plot_" + s_model_name + '.png', dpi=600)

    # save the shap values - TEST DATA
    df_shap_values = pd.DataFrame(shap_values, columns=df_X_to_plot.columns)
    df_shap_values.to_csv(dir_to_save_plot + "/SHAP_values_" + s_model_name + '.csv', index=False)

    # save the explainer

    # Save the SHAP explainer object to disk
    explainer_path = dir_to_save_plot + '/SHAP_explainer_' + s_model_name + '.pkl'
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)

    return shap_values, explainer


def r_score(x, y):
    """Compute the Pearson correlation coefficient (R) between two arrays.

    Parameters
    ----------
    x, y : array-like
        Arrays of values to correlate.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    return stats.pearsonr(x.ravel(), y.ravel())[0]


def r2_score(x, y):
    """Compute the squared Pearson correlation coefficient (R^2) between two arrays.

    Parameters
    ----------
    x, y : array-like
        Arrays of values to correlate.

    Returns
    -------
    float
        Squared Pearson correlation coefficient.
    """
    return stats.pearsonr(x.ravel(), y.ravel())[0] ** 2


def shapley_feature_ranking(shap_values, X):
    """Rank features by mean absolute SHAP value (descending importance).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix, shape (n_samples, n_features).
    X : pd.DataFrame
        Feature matrix (used for column names).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``'features'`` and ``'importance'``, sorted
        from most to least important.
    """
    feature_order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(np.abs(shap_values), axis=0)[i] for i in feature_order
            ][::-1],
        }
    )

