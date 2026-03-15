import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import seaborn as sns

# ==========================================================
# METRICS
# ==========================================================
def compute_metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def bootstrap_lightgbm_forward_selection(
    df,
    target,
    n_bootstrap,
    n_max_variables,
    metric_to_optimize,
    hyperparameters
):

    """
    Perform bootstrap-based forward feature selection using LightGBM.

    Methodology
    -----------
    • Bootstrap sampling with replacement.
    • True Out-of-Bag (OOB) validation.
    • Greedy forward selection.
    • Metric evaluated strictly on OOB samples.
    • Feature ranking stability can be analyzed across bootstraps.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset including target column.
    target : str
        Name of target column (binary classification expected).
    n_bootstrap : int
        Number of bootstrap resamples.
    n_max_variables : int
        Maximum number of variables to select.
    metric_to_optimize : str
        Metric key returned by compute_metrics.
    hyperparameters : dict
        LightGBM hyperparameters.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing:
        - "variables": selected features per bootstrap
        - one DataFrame per metric
          (rows = number of variables, columns = bootstrap iteration)
    """
    
    X_full = df.drop(columns=[target])
    y_full = df[target].astype(int)

    n_samples = len(df)

    # Estruturas finais
    results_metrics = {
        "auc_roc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    results_variables = []

    for b in tqdm(range(n_bootstrap)):

        rng = np.random.default_rng(b)

        # ======================================================
        # BOOTSTRAP ESTATISTICAMENTE CORRETO (OOB real)
        # ======================================================
        bootstrap_idx = rng.integers(0, n_samples, n_samples)

        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bootstrap_idx] = False

        if oob_mask.sum() == 0:
            continue  # For rare cases

        X_train = X_full.iloc[bootstrap_idx]
        y_train = y_full.iloc[bootstrap_idx]

        X_val = X_full.iloc[oob_mask]
        y_val = y_full.iloc[oob_mask]

        # ======================================================
        # FORWARD SELECTION
        # ======================================================
        selected_features = []
        remaining_features = list(X_full.columns)

        metrics_history = []

        for _ in range(min(n_max_variables, len(remaining_features))):

            best_feature = None
            best_score = -np.inf

            for feature in remaining_features:

                trial_features = selected_features + [feature]

                model_params = hyperparameters.copy()
                model_params["random_state"] = b

                model = lgb.LGBMClassifier(**model_params)

                model.fit(X_train[trial_features], y_train)

                y_proba = model.predict_proba(X_val[trial_features])[:, 1]

                metrics = compute_metrics(y_val, y_proba)

                score = metrics[metric_to_optimize]

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_metrics = metrics

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

            metrics_history.append(best_metrics)

        # Salva resultados do bootstrap
        results_variables.append(selected_features)

        for metric_name in results_metrics.keys():
            results_metrics[metric_name].append(
                [step[metric_name] for step in metrics_history]
            )

    # ======================================================
    # OUTPUT
    # ======================================================
    metrics_df = {
        metric: pd.DataFrame(values)
        for metric, values in results_metrics.items()
    }

    variables_df = pd.DataFrame(results_variables)

    # Transpose so each row is a variable and each column is a bootstrap
    dict_return = {
        "variables": variables_df,
        **metrics_df
    }
    for key in dict_return:
        dict_return[key] = dict_return[key].T
    return dict_return



def performance_forward_selection_boxplot(df_metric, metric_name):

    """
    Creates a boxplot showing the distribution of a performance metric 
    across different numbers of selected variables, for multiple bootstrap iterations.

    This visualization helps to analyze how model performance behaves 
    as the number of features increases/decreases, including stability 
    (spread across bootstraps) and potential overfitting/underfitting patterns.

    Parameters
    ----------
    df_metric : pd.DataFrame
        DataFrame where:
        - Index = number of variables (0, 1, 2, ...) 
        - Columns = different bootstrap iterations
        - Values = performance metric (e.g. AUC, F1, RMSE, etc.)
    metric_name : str
        Name of the metric being plotted (used in axis label and title)
    figsize : tuple, optional
        Figure size (width, height), by default (15, 6)
    title : str or None, optional
        Custom title for the plot. If None, a default title is generated.
    xlabel : str, optional
        Label for the x-axis, by default "Number of Variables"
    ylabel : str or None, optional
        Label for the y-axis. If None, uses metric_name capitalized.
    palette : str, optional
        Color palette name for seaborn, by default "viridis"

    Returns
    -------
    None
        Displays the plot (does not return the figure/axis)

    Examples
    --------
    >>> performance_boxplot(df_auc_results, "AUC", figsize=(16, 7), palette="magma")
    >>> performance_boxplot(df_rmse, "RMSE", title="Model RMSE vs Number of Features")
    """
    
    df_aux = df_metric.copy()

    # The index is the number of variables
    df_aux = df_aux.reset_index()
    df_aux.rename(columns={"index": "n_variables"}, inplace=True)

    # initialize count for variables
    df_aux["n_variables"] += 1

    # Long format
    df_long = df_aux.melt(
        id_vars="n_variables",
        var_name="bootstrap",
        value_name=metric_name
    )

    # =============================
    # Plot
    # =============================
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(rc={'figure.figsize': (15, 6)})
    sns.set_style("darkgrid")

    meanprops = dict(color='black', linewidth=2)

    ax = sns.boxplot(
        data=df_long,
        x="n_variables",
        y=metric_name,
        showmeans=True,
        meanline=True,
        meanprops=meanprops
    )

    ax.set_title("Performance by Number of Variables")

    plt.show()


def variable_frequency_forward_selection(df, n_bootstraps):
    """
    Creates a heatmap showing the frequency (proportion) with which each variable
    was selected in models with different numbers of variables.
    """

    df_aux = df.copy()

    # Convert the data into counts per variable for each n_variables combination
    df_variables_heatmap = (
        df_aux.iloc[:, :-1]
        .apply(pd.Series.value_counts, axis=1)
        .fillna(0)
    )

    # Normalize by number of bootstraps → convert counts into proportions
    df_variables_heatmap = df_variables_heatmap / n_bootstraps

    # Cumulative frequency (forward selection effect)
    df_variables_heatmap = df_variables_heatmap.cumsum()

    # Remove zeros
    df_variables_heatmap = df_variables_heatmap.replace({0: np.nan})

    # Ensure counting starts at 1
    df_variables_heatmap["n_variables"] = range(1, len(df_variables_heatmap) + 1)

    # Set index
    df_variables_heatmap = df_variables_heatmap.set_index("n_variables")

    # -------------------------------------------------
    # ROW ORDERING ALGORITHM (greedy by column)
    # -------------------------------------------------
    df_order = df_variables_heatmap.T.copy()

    ordered_rows = []
    remaining_rows = list(df_order.index)

    for col in df_order.columns:

        if not remaining_rows:
            break

        # choose row with largest value in this column
        best_row = df_order.loc[remaining_rows, col].idxmax()

        ordered_rows.append(best_row)
        remaining_rows.remove(best_row)

    # append remaining rows if any
    ordered_rows.extend(remaining_rows)

    df_order = df_order.loc[ordered_rows]

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    size_x = 8
    size_y = 8

    plt.figure(figsize=(size_x + 5, size_y + 13))

    sns.set(font_scale=0.7)

    with sns.axes_style("white"):
        ax = sns.heatmap(
            df_order,
            linewidths=0.2,
            annot=True,
            fmt=".1f",
            cmap="seismic",
            vmin=-1,
            vmax=1
        )

    plt.title("Most used variables")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()
    plt.close()



# Display the top-k variables using the same ordering used in the
# variable_frequency_forward_selection heatmap.
def _compute_forward_selection_order(df_variables_heatmap):
    """
    Internal function to compute the greedy ordering used in the heatmap.
    """

    df_order = df_variables_heatmap.T.copy()

    ordered_rows = []
    remaining_rows = list(df_order.index)

    for col in df_order.columns:

        if not remaining_rows:
            break

        best_row = df_order.loc[remaining_rows, col].idxmax()

        ordered_rows.append(best_row)
        remaining_rows.remove(best_row)

    ordered_rows.extend(remaining_rows)

    return ordered_rows
def top_k_forward_selection_variables(df, n_bootstraps, k=10):
    """
    Return the top-k variables using the same ordering used in the
    variable_frequency_forward_selection heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        result_bootstrap["variables"]

    n_bootstraps : int
        Number of bootstrap iterations.

    k : int, default=10
        Number of variables to return.

    return_df : bool, default=False
        If True, returns a DataFrame with variable ranks.

    Returns
    -------
    list or pd.DataFrame
        List of top-k variables or a DataFrame with ranking.
    """

    df_aux = df.copy()

    df_variables_heatmap = (
        df_aux.iloc[:, :-1]
        .apply(pd.Series.value_counts, axis=1)
        .fillna(0)
    )

    df_variables_heatmap = df_variables_heatmap / n_bootstraps
    df_variables_heatmap = df_variables_heatmap.cumsum()
    df_variables_heatmap = df_variables_heatmap.replace({0: np.nan})

    df_variables_heatmap["n_variables"] = range(1, len(df_variables_heatmap) + 1)
    df_variables_heatmap = df_variables_heatmap.set_index("n_variables")

    ordered_rows = _compute_forward_selection_order(df_variables_heatmap)

    top_k = ordered_rows[:k]

    return top_k



# Returns the k variables that produced the best performancefor a given metric.
def top_k_variables_by_forward_selection_boxplot(result_bootstrap, k, metric):
    """
    Returns the k variables that produced the best performance
    for a given metric.

    Parameters
    ----------
    result_bootstrap : dict
        Dictionary containing:
        - "variables"
        - performance metrics (e.g. "auc_roc", "accuracy")

    k : int
        Number of variables in the model

    metric : str
        Performance metric key inside result_bootstrap

    Returns
    -------
    variables : list
        List with the k variables

    performance : float
        Best performance obtained
    """

    df_vars = result_bootstrap["variables"]
    df_metric = result_bootstrap[metric]

    row = k - 1

    # performance row
    perf_row = df_metric.loc[row]

    # maximum performance
    performance = perf_row.max()

    # bootstrap column where max occurs
    best_col = perf_row.idxmax()

    # corresponding variables
    variables = df_vars.loc[:row, best_col].tolist()

    return variables, performance
