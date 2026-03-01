import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Any, Optional, Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def bootstrap_tree_binning_auc_analysis(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature: str,
    target: str,
    max_leaf_nodes_max: int = 7,
    n_bootstrap: int = 30,
    min_samples_leaf: float = 0.10,
    random_state: int = 42,
    plot: bool = True,
    ax: Optional[plt.Axes] = None
) -> Dict[str, Any]:
    """
    Evaluate stability and predictive performance of supervised univariate
    tree-based binning via bootstrap resampling.

    The function evaluates tree complexity from 2 to max_leaf_nodes_max
    and computes validation ROC AUC for each bootstrap resample.

    Returns detailed bootstrap distributions and summary statistics.
    """

    # -------------------------
    # Input validation
    # -------------------------
    for df_name, df in [("df_train", df_train), ("df_val", df_val)]:
        if feature not in df.columns:
            raise ValueError(f"{feature} not found in {df_name}.")
        if target not in df.columns:
            raise ValueError(f"{target} not found in {df_name}.")

    if not np.issubdtype(df_train[feature].dtype, np.number):
        raise ValueError("Feature must be numeric.")

    if max_leaf_nodes_max < 2:
        raise ValueError("max_leaf_nodes_max must be >= 2.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")

    # Clean datasets
    df_train_clean = df_train[[feature, target]].dropna()
    df_val_clean = df_val[[feature, target]].dropna()

    if df_train_clean[target].nunique() != 2:
        raise ValueError("Training target must be binary.")

    if df_val_clean[target].nunique() != 2:
        raise ValueError("Validation target must be binary.")

    X_val = df_val_clean[[feature]]
    y_val = df_val_clean[target]

    # -------------------------
    # Containers
    # -------------------------
    auc_results: Dict[int, list] = {
        k: [] for k in range(2, max_leaf_nodes_max + 1)
    }

    splits_dict: Dict[int, list] = {
        k: [] for k in range(2, max_leaf_nodes_max + 1)
    }

    rng = np.random.RandomState(random_state)

    # -------------------------
    # Bootstrap evaluation
    # -------------------------
    for max_leaf in range(2, max_leaf_nodes_max + 1):

        for _ in range(n_bootstrap):

            df_boot = df_train_clean.sample(
                frac=1,
                replace=True,
                random_state=rng.randint(0, 1_000_000)
            )

            # Skip if bootstrap collapses to single class
            if df_boot[target].nunique() < 2:
                continue

            X_boot = df_boot[[feature]]
            y_boot = df_boot[target]

            tree = DecisionTreeClassifier(
                max_leaf_nodes=max_leaf,
                min_samples_leaf=min_samples_leaf,
                random_state=rng.randint(0, 1_000_000)
            )

            tree.fit(X_boot, y_boot)

            # Validation AUC
            try:
                y_pred_val = tree.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_val)
                auc_results[max_leaf].append(auc)
            except ValueError:
                continue

            # Threshold extraction
            thresholds = tree.tree_.threshold
            valid_thresholds = np.sort(thresholds[thresholds > -2])
            splits_dict[max_leaf].append(valid_thresholds)

    # -------------------------
    # Summary statistics
    # -------------------------
    auc_summary = {
        k: {
            "mean_auc": np.mean(v) if len(v) > 0 else np.nan,
            "std_auc": np.std(v) if len(v) > 0 else np.nan,
            "n_valid_bootstrap": len(v)
        }
        for k, v in auc_results.items()
    }

    # -------------------------
    # Plot
    # -------------------------
    fig: Optional[plt.Figure] = None

    if plot:

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        data = [auc_results[k] for k in sorted(auc_results)]
        positions = sorted(auc_results)

        ax.boxplot(data, positions=positions)
        ax.set_xlabel("max_leaf_nodes")
        ax.set_ylabel("Validation ROC AUC")
        ax.set_title(
            f"Bootstrap ROC AUC by Tree Complexity — Feature: {feature}"
        )
        ax.grid(False)

        fig.tight_layout()

    return {
        "auc_results": auc_results,
        "auc_summary": auc_summary,
        "splits_dict": splits_dict,
        "figure": fig,
        "axis": ax if plot else None
    }




def tree_supervised_binning(
    df: pd.DataFrame,
    feature: str,
    target: str,
    max_leaf_nodes: int,
    min_samples_leaf: Union[int, float] = 0.10,
    random_state: int = 1,
    plot: bool = True,
    annotate_counts: bool = True
) -> Dict[str, Any]:
    """
    Perform supervised univariate binning using a Decision Tree classifier
    and optionally plot target mean per bin with observation counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing feature and target.
    feature : str
        Name of numeric feature to bin.
    target : str
        Name of binary target variable.
    max_leaf_nodes : int
        Maximum number of leaf nodes.
    min_samples_leaf : int or float, default=0.10
        Minimum samples per leaf (absolute or fraction).
    random_state : int, default=1
        Random seed.
    plot : bool, default=True
        Whether to generate the plot.
    annotate_counts : bool, default=True
        Whether to annotate observation counts on bars.

    Returns
    -------
    dict
        {
            "thresholds": np.ndarray,
            "bin_summary": pd.DataFrame
        }

    Raises
    ------
    ValueError
        If validation fails.
    """

    # -------------------------
    # Validation
    # -------------------------
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found.")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    if not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 2:
        raise ValueError("max_leaf_nodes must be integer >= 2.")

    df_clean = df[[feature, target]].dropna()

    if df_clean.empty:
        raise ValueError("No data available after dropping NA.")

    if not np.issubdtype(df_clean[feature].dtype, np.number):
        raise ValueError("Feature must be numeric.")

    if df_clean[target].nunique() != 2:
        raise ValueError("Target must be binary.")

    X = df_clean[[feature]]
    y = df_clean[target]

    # -------------------------
    # Fit tree
    # -------------------------
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    tree.fit(X, y)

    thresholds = np.sort(
        tree.tree_.threshold[tree.tree_.threshold > -2]
    )

    # -------------------------
    # Create bins without mutating original df
    # -------------------------
    bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
    bin_series = pd.cut(df_clean[feature], bins=bins)

    summary = (
        pd.DataFrame({
            "bin": bin_series,
            target: df_clean[target]
        })
        .groupby("bin", observed=True)[target]
        .agg(count="count", target_rate="mean")
        .reset_index()
    )

    # -------------------------
    # Plot
    # -------------------------
    if plot:

        fig, ax = plt.subplots(figsize=(10, 5))
    
        x_positions = np.arange(len(summary))
        bars = ax.bar(
            x_positions,
            summary["target_rate"].values
        )
    
        ax.set_ylabel("Mean Target")
        ax.set_xlabel("Bins")
        ax.set_title(f"Target Mean by Tree Bins — {feature}")
    
        # Define ticks corretamente
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            summary["bin"].astype(str),
            rotation=45,
            ha="right"
        )
    
        if annotate_counts:
            for bar, count in zip(bars, summary["count"]):
                height = bar.get_height()
                ax.annotate(
                    f"n={count}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )
    
        plt.tight_layout()
        plt.show()

    return {
        "thresholds": thresholds,
        "bin_summary": summary
    }