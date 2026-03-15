# Bootstrap ML Diagnostics

A lightweight toolkit for **statistically robust model diagnostics** using **bootstrap resampling**, with utilities for:

* supervised tree binning
* bootstrap-based feature selection
* model stability analysis
* hyperparameter sensitivity analysis

The library focuses on **reducing overfitting and improving model interpretability** through **bootstrap distributions rather than single-point estimates**.

---

# Installation

```bash
pip install maxwailab 
```

or

```bash
pip install git+https://github.com/MaxWienandts/maxailab.git
```

---

# Core Philosophy

Most ML workflows rely on **single train/validation splits**.

This library instead uses **bootstrap resampling** to estimate:

* performance **distributions**
* feature **selection stability**
* hyperparameter **robustness**

Benefits:

* reduces variance from a single split
* identifies unstable variables
* provides confidence intervals for model performance

---

# Workflow Overview

Typical modeling workflow using this library:

```
1️⃣ Supervised binning (optional)

tree_supervised_binning
bootstrap_tree_binning_auc_analysis


2️⃣ Feature selection

bootstrap_lightgbm_forward_selection


3️⃣ Diagnostics

performance_forward_selection_boxplot
variable_frequency_forward_selection


4️⃣ Extract best variables

top_k_forward_selection_variables
top_k_variables_by_forward_selection_boxplot


5️⃣ Hyperparameter analysis

lightgbm_hyperparameter_auc_curve_bootstrap
```

---

# Example Workflow

```python
import maxwailab

# --------------------------------
# Forward Selection with Bootstrap
# --------------------------------

result_bootstrap = maxwailab.bootstrap_lightgbm_forward_selection(
    df=data,
    target="target",
    n_bootstrap=30,
    n_max_variables=15,
    metric_to_optimize="auc_roc",
    hyperparameters=lgb_params
)
```

---

## Analyze performance stability

```python
bml.performance_forward_selection_boxplot(
    result_bootstrap["auc_roc"],
    "AUC"
)
```

This visualizes how performance behaves as variables are added.

---

## Variable selection stability

```python
maxwailab.variable_frequency_forward_selection(
    result_bootstrap["variables"],
    n_bootstraps=30
)
```

Heatmap showing **how frequently variables appear in models of different sizes**.

---

## Extract best variables

### Based on selection frequency

```python
maxwailab.top_k_forward_selection_variables(
    result_bootstrap["variables"],
    n_bootstraps=30,
    k=10
)
```

### Based on best model performance

```python
variables, auc = maxwailab.top_k_variables_by_forward_selection_boxplot(
    result_bootstrap,
    k=6,
    metric="auc_roc"
)
```

---

# Tree-based Supervised Binning

Supervised binning using decision trees.

```python
from maxwailab import tree_supervised_binning

tree_supervised_binning(
    df=data,
    feature="age",
    target="target",
    max_leaf_nodes=5
)
```

---

## Bootstrap Binning Stability

```python
bootstrap_tree_binning_auc_analysis(
    df_train,
    df_val,
    feature="age",
    target="target"
)
```

Evaluates how **binning performance varies across bootstrap samples**.

---

# Hyperparameter Sensitivity Analysis

Evaluate how model performance reacts to hyperparameter changes.

```python
lightgbm_hyperparameter_auc_curve_bootstrap(
    X_train,
    y_train,
    X_val,
    y_val,
    hyperparameters=lgb_params,
    hyperparameter_name="num_leaves",
    hyperparameter_values=[5,10,20,40],
    n_bootstrap=50
)
```

Bootstrap is applied **only to the training set** while keeping validation **fixed (out-of-time)**.

---

# Example Output

The library produces:

* performance **distributions**
* **boxplots**
* **stability heatmaps**
* **hyperparameter sensitivity curves**

These diagnostics help detect:

* overfitting
* unstable features
* fragile hyperparameters

---

# Module Structure

```
maxwailab
│
├── binning
│   ├── tree_supervised_binning
│   └── bootstrap_tree_binning_auc_analysis
│
├── feature_selection
│   ├── bootstrap_lightgbm_forward_selection
│   ├── performance_forward_selection_boxplot
│   ├── variable_frequency_forward_selection
│   ├── top_k_forward_selection_variables
│   └── top_k_variables_by_forward_selection_boxplot
│
└── hyperparameter_analysis
    └── lightgbm_hyperparameter_auc_curve_bootstrap
```

---

# When to Use This Library

This library is particularly useful for:

* **credit risk models**
* **tabular ML problems**
* **high-stakes predictive modeling**
* **interpretable ML workflows**

---

# License

MIT License