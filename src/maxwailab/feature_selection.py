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
# MÉTRICAS
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
# FUNÇÃO PRINCIPAL
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
            continue  # caso raríssimo

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
    # ORGANIZA OUTPUT
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
    
    # Garante cópia
    df_aux = df_metric.copy()

    # Índice representa número de variáveis (step)
    df_aux = df_aux.reset_index()
    df_aux.rename(columns={"index": "n_variables"}, inplace=True)

    # Começar contagem em 1
    df_aux["n_variables"] += 1

    # Converte para formato long
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
    Cria um heatmap mostrando a frequência (proporção) com que cada variável
    foi selecionada em cada modelo com diferentes quantidades de variáveis.
    
    Parâmetros:
    - df: DataFrame com colunas: variáveis + 'n_variables' + 'modelo' (ou similar)
    - n_bootstraps: número total de bootstraps realizados (para calcular proporção)
    """
    df_aux = df.copy()
    
    # Aqui transformamos em contagem por variável por combinação de n_variables
    df_variables_heatmap = (
        df_aux.iloc[:, :-1]                 
        .apply(pd.Series.value_counts, axis = 1)
        .fillna(0)
    )
    
    # Normaliza pela quantidade de bootstraps → vira proporção
    df_variables_heatmap = df_variables_heatmap / n_bootstraps

    df_variables_heatmap = df_variables_heatmap.cumsum()
    # Remove possíveis NaN remanescentes
    df_variables_heatmap = df_variables_heatmap.replace({0: np.nan})

    # Garantir contagem iniciando em 1
    df_variables_heatmap["n_variables"] = range(1, len(df_variables_heatmap) + 1)
    
    # Tornar n_variables o index
    df_variables_heatmap = df_variables_heatmap.set_index("n_variables")
    
    # Configuração do gráfico
    size_x = 8
    size_y = 8
    
    plt.figure(figsize=(size_x + 5, size_y + 13))
    
    sns.set(font_scale=0.7)
    
    with sns.axes_style('white'):
        ax = sns.heatmap(
            df_variables_heatmap.T,                    # transposto para variáveis nas linhas
            linewidths=0.2,
            annot=True,
            fmt='.1f',                                 # formato original era '.1f' em alguns trechos
            cmap='seismic',
            vmin=-1,
            vmax=1
        )
    
    plt.title('Most used variables')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.show()
    plt.close()