"""
Combined structural + whole‑brain ML pipeline for SNHL classification.

Steps:
1. Load and merge features from two CSV files.
2. Exploratory visualization (3D surface, correlation matrices).
3. Train/test split with stratification.
4. KNN imputation and outlier removal.
5. Correlation‑based feature pruning.
6. Scaling and RFE with LightGBM.
7. Bayesian hyperparameter search for multiple classifiers.
8. Cross‑validated performance estimation and learning curves.
9. Threshold optimization without data leakage.
10. Final test evaluation, plots, and Word report export.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from docx import Document
from docx.shared import Inches


# -------------------------------------------------------------------------
# 1. Paths and basic configuration
# -------------------------------------------------------------------------

STRUCTURAL_CSV_PATH = r"D:\snhl\structural_output.csv"
WHOLEBRAIN_CSV_PATH = r"D:\snhl\Whole_output.csv"

CORR_ALL_FEATURES_PATH = r"D:\corr_combined_all.png"
CORR_SELECTED_FEATURES_PATH = r"D:\corr_combined_selected.png"
WORD_TABLE_PATH = r"D:\table_combined_selected_features.docx"
WORD_RESULTS_PATH = r"D:\model_results_combined_leakfree.docx"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 10
N_FEATURES_RFE = 8
CORRELATION_THRESHOLD = 0.8


# -------------------------------------------------------------------------
# 2. Utility functions
# -------------------------------------------------------------------------

def plot_3d_surface_from_dataframe(df_numeric, title="3D Surface Plot"):
    """
    Plot a 3D surface where:
    - X axis: feature index
    - Y axis: subject index
    - Z axis: feature value.
    """
    numeric_data = df_numeric.select_dtypes(include=[np.number])
    x_axis = np.arange(numeric_data.shape[1])  # feature indices
    y_axis = np.arange(numeric_data.shape[0])  # subject indices
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    z_values = numeric_data.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(x_mesh, y_mesh, z_values, cmap="viridis")

    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Feature Value")
    plt.style.use("default")

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Subject Index")
    ax.set_zlabel("Feature Value")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def remove_extreme_outliers(
    data_frame, feature_threshold=0.7, subject_threshold=0.1
):
    """
    Remove subjects that have a high proportion of outlier features.

    Outliers per feature are defined via IQR (1.5 * IQR rule).
    A subject is removed if:
        (#outlier_features / total_features) > feature_threshold
    and the overall proportion of such subjects is < subject_threshold.
    """
    df = data_frame.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    total_features = len(numeric_cols)

    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

    df["outlier_count"] = outlier_mask.sum(axis=1)
    df["outlier_subject"] = (df["outlier_count"] / total_features) > feature_threshold

    num_outlier_subjects = df["outlier_subject"].sum()
    total_subjects = len(df)

    if total_subjects > 0 and (num_outlier_subjects / total_subjects) < subject_threshold:
        df = df[~df["outlier_subject"]]

    df = df.drop(columns=["outlier_count", "outlier_subject"])
    return df


def plot_correlation_matrix(data, title=None, save_path=None):
    """
    Plot a compact correlation matrix heatmap for the numeric part of a DataFrame.
    Constant and NaN-only columns are removed.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.loc[:, numeric_data.nunique() > 1]
    numeric_data = numeric_data.dropna(axis=0)

    corr_matrix = numeric_data.corr()
    if corr_matrix.empty or corr_matrix.shape[0] != corr_matrix.shape[1]:
        print("Correlation matrix is empty or invalid.")
        return

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        corr_matrix,
        cmap="viridis",
        center=0,
        vmin=-1,
        vmax=1,
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        linecolor="white",
        fmt=".2f",
        annot_kws={"size": 0.4},
        cbar_kws={"shrink": 0.6, "ticks": [-1, -0.5, 0, 0.5, 1]},
    )
    plt.title(title or "Correlation Matrix", fontsize=10)
    plt.style.use("default")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()


def plot_learning_curve(
    estimator,
    x_data,
    y_data,
    cv,
    scoring="balanced_accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
    model_name="Model",
):
    """
    Plot learning curves for a given model and dataset using cross-validation.
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, x_data, y_data, cv=cv, scoring=scoring, train_sizes=train_sizes
    )

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, train_scores_mean, "o-", label="Training Accuracy")
    plt.plot(train_sizes_abs, test_scores_mean, "o-", label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Balanced Accuracy")
    plt.title(f"Learning Curve: {model_name}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_best_threshold(estimator, x_val, y_val):
    """
    For probabilistic classifiers, find the classification threshold
    that maximizes balanced accuracy on a validation set.
    Returns 0.5 for non-probabilistic models.
    """
    if not hasattr(estimator, "predict_proba"):
        return 0.5

    y_proba = estimator.predict_proba(x_val)[:, 1]
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_thresh = 0.5
    best_bal_acc = 0.0

    for threshold in candidate_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = threshold

    return best_thresh


def estimate_threshold_via_cv(estimator, x_data, y_data, cv_splits=5):
    """
    Estimate the optimal classification threshold using internal
    cross-validation on the training data (no test leakage).
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    thresholds = []

    for train_idx, val_idx in cv.split(x_data, y_data):
        x_train_cv, x_val_cv = x_data[train_idx], x_data[val_idx]
        y_train_cv, y_val_cv = y_data[train_idx], y_data[val_idx]

        estimator.fit(x_train_cv, y_train_cv)
        threshold = find_best_threshold(estimator, x_val_cv, y_val_cv)
        thresholds.append(threshold)

    return float(np.mean(thresholds)) if len(thresholds) > 0 else 0.5


def plot_model_evaluation(
    estimator,
    x_data,
    y_true,
    model_name,
    feature_names=None,
    threshold=0.5,
):
    """
    Create evaluation plots:
    - Confusion matrix
    - ROC curve
    - Precision–Recall curve
    - Feature importance (if available)

    Saves the figure as '{model_name}_evaluation.png' and returns predictions.
    """
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(x_data)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_proba = None
        y_pred = estimator.predict(x_data)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Model: {model_name} (Threshold={threshold:.2f})", fontsize=16)

    # Confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        confusion,
        cmap="winter",
        annot=True,
        fmt="d",
        xticklabels=["Healthy", "Patients"],
        yticklabels=["Healthy", "Patients"],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")

        # Precision–Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        axes[2].plot(recall, precision, label=f"AP = {avg_precision:.2f}")
        axes[2].set_title("Precision–Recall Curve")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].legend(loc="lower left")
    else:
        axes[1].axis("off")
        axes[2].axis("off")

    # Feature importance
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        sorted_names = np.array(feature_names)[sorted_indices]
        axes[3].barh(sorted_names[::-1], importances[sorted_indices][::-1])
        axes[3].set_title("Feature Importance")
        axes[3].set_xlabel("Importance")
    else:
        axes[3].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{model_name}_evaluation.png", dpi=300)
    plt.show()

    return y_pred


# -------------------------------------------------------------------------
# 3. Data loading and merging
# -------------------------------------------------------------------------

# Load structural and whole‑brain feature CSV files
structural_df = pd.read_csv(STRUCTURAL_CSV_PATH)
wholebrain_df = pd.read_csv(WHOLEBRAIN_CSV_PATH)

# Concatenate along columns (features)
combined_df = pd.concat([wholebrain_df, structural_df], axis=1)

# Drop duplicated or non-feature columns if present
for col_to_drop in ["0 Subject/Sample #", "label"]:
    if col_to_drop in combined_df.columns:
        combined_df = combined_df.drop(columns=col_to_drop)

# -------------------------------------------------------------------------
# 4. Exploratory visualization
# -------------------------------------------------------------------------

plot_3d_surface_from_dataframe(
    combined_df.select_dtypes(include=[np.number]),
    title="3D Surface Plot of Combined Features",
)

# Add a synthetic label based on index (0–29: control, 30+: patient)
combined_df["label"] = combined_df.index.to_series().apply(
    lambda idx: 0 if int(idx) < 30 else 1
)

# Separate features and labels
feature_df = combined_df.drop(columns="label")
labels = combined_df["label"].values

# Train–test split with stratification
x_train, x_test, y_train, y_test = train_test_split(
    feature_df,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels,
)

y_train_series = pd.Series(y_train, name="label")

# -------------------------------------------------------------------------
# 5. Imputation, outlier removal, correlation-based pruning
# -------------------------------------------------------------------------

# KNN imputation (fit on training only)
knn_imputer = KNNImputer(n_neighbors=5)
x_train_imputed = knn_imputer.fit_transform(x_train)
x_test_imputed = knn_imputer.transform(x_test)

x_train_imputed_df = pd.DataFrame(
    x_train_imputed,
    columns=x_train.columns,
    index=x_train.index,
)
x_test_imputed_df = pd.DataFrame(
    x_test_imputed,
    columns=x_train.columns,
    index=x_test.index,
)

# Remove extreme outlier subjects from training data only
x_train_clean = remove_extreme_outliers(x_train_imputed_df)
x_test_clean = x_test_imputed_df[x_train_clean.columns]

# Correlation-based feature removal
correlation_matrix = x_train_clean.corr()
label_correlation = x_train_clean.corrwith(y_train_series).abs()

high_correlation_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
            feat_i = correlation_matrix.columns[i]
            feat_j = correlation_matrix.columns[j]
            drop_feature = feat_i if label_correlation[feat_i] < label_correlation[feat_j] else feat_j
            high_correlation_features.add(drop_feature)

x_train_pruned = x_train_clean.drop(columns=high_correlation_features)
x_test_pruned = x_test_clean.drop(columns=high_correlation_features)

# Correlation matrix before RFE (optional)
plot_correlation_matrix(
    feature_df,
    title="Correlation Matrix: All Combined Features",
    save_path=CORR_ALL_FEATURES_PATH,
)

# -------------------------------------------------------------------------
# 6. Scaling and RFE (LightGBM)
# -------------------------------------------------------------------------

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_pruned)
x_test_scaled = scaler.transform(x_test_pruned)

lgbm_estimator = LGBMClassifier(
    n_estimators=200,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
)

rfe_selector = RFE(
    estimator=lgbm_estimator,
    n_features_to_select=N_FEATURES_RFE,
    step=1,
)
rfe_selector.fit(x_train_scaled, y_train_series)

x_train_rfe = rfe_selector.transform(x_train_scaled)
x_test_rfe = rfe_selector.transform(x_test_scaled)

selected_feature_names = list(x_train_pruned.columns[rfe_selector.get_support()])
print("Selected Features:", selected_feature_names)

# Correlation matrix of selected features
selected_train_df = pd.DataFrame(x_train_rfe, columns=selected_feature_names)
plot_correlation_matrix(
    selected_train_df,
    title="Correlation Matrix: Selected Combined Features",
    save_path=CORR_SELECTED_FEATURES_PATH,
)

# -------------------------------------------------------------------------
# 7. Optionally save selected feature names in a Word table
# -------------------------------------------------------------------------

selected_features_for_doc = [[name] for name in selected_feature_names]

feature_doc = Document()
feature_doc.add_heading("Selected Combined Features", level=1)
feature_table = feature_doc.add_table(
    rows=len(selected_features_for_doc),
    cols=1,
)

for row_idx, row_values in enumerate(selected_features_for_doc):
    for col_idx, cell_value in enumerate(row_values):
        feature_table.cell(row_idx, col_idx).text = str(cell_value)

feature_doc.save(WORD_TABLE_PATH)

# -------------------------------------------------------------------------
# 8. Model definitions and hyperparameter spaces
# -------------------------------------------------------------------------

models = {
    "RandomForest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "ExtraTrees": ExtraTreesClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "NuSVC": NuSVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(
        solver="liblinear",
        random_state=RANDOM_STATE,
    ),
    "CalibratedRidge": CalibratedClassifierCV(
        estimator=RidgeClassifier(random_state=RANDOM_STATE),
        method="sigmoid",
        cv=5,
    ),
    "CalibratedLinearSVC": CalibratedClassifierCV(
        estimator=LinearSVC(random_state=RANDOM_STATE, dual=False),
        method="sigmoid",
        cv=5,
    ),
    "QDA": QuadraticDiscriminantAnalysis(),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
}

param_spaces = {
    "RandomForest": {
        "clf__n_estimators": Integer(50, 300),
        "clf__max_depth": Integer(3, 20),
        "clf__min_samples_split": Integer(2, 20),
        "clf__min_samples_leaf": Integer(1, 10),
        "clf__max_features": Categorical(["sqrt", "log2", None]),
        "clf__bootstrap": Categorical([True, False]),
        "clf__criterion": Categorical(["gini", "entropy", "log_loss"]),
    },
    "ExtraTrees": {
        "clf__n_estimators": Integer(50, 300),
        "clf__max_depth": Integer(3, 20),
        "clf__min_samples_split": Integer(2, 20),
        "clf__min_samples_leaf": Integer(1, 10),
        "clf__max_features": Categorical(["sqrt", "log2", None]),
    },
    "GradientBoosting": {
        "clf__n_estimators": Integer(200, 1200),
        "clf__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
        "clf__max_depth": Integer(3, 6),
        "clf__subsample": Real(0.6, 0.9),
        "clf__loss": Categorical(["log_loss"]),
        "clf__min_samples_split": Integer(10, 100),
        "clf__min_samples_leaf": Integer(20, 200),
        "clf__max_features": Real(0.2, 0.8),
        "clf__max_leaf_nodes": Integer(10, 200),
    },
    "HistGradientBoosting": {
        "clf__learning_rate": Real(0.05, 0.2, prior="log-uniform"),
        "clf__max_iter": Integer(30, 700),
        "clf__max_depth": Integer(5, 9),
        "clf__min_samples_leaf": Integer(10, 60),
        "clf__l2_regularization": Real(0.05, 1.5),
        "clf__max_bins": Integer(128, 255),
        "clf__early_stopping": Categorical([True]),
    },
    "XGBoost": {
        "clf__n_estimators": Integer(50, 300),
        "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "clf__max_depth": Integer(3, 12),
        "clf__subsample": Real(0.6, 1.0),
        "clf__colsample_bytree": Real(0.6, 1.0),
        "clf__reg_alpha": Real(1e-6, 10.0, prior="log-uniform"),
        "clf__reg_lambda": Real(1e-6, 10.0, prior="log-uniform"),
    },
    "KNN": {
        "clf__n_neighbors": Integer(3, 20),
        "clf__weights": Categorical(["uniform", "distance"]),
        "clf__p": Integer(1, 2),
        "clf__algorithm": Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
    },
    "SVM": {
        "clf__C": Real(0.01, 10.0, prior="log-uniform"),
        "clf__kernel": Categorical(["linear", "rbf", "poly", "sigmoid"]),
        "clf__gamma": Categorical(["scale", "auto"]),
        "clf__degree": Categorical([2, 3, 4, 5, 6]),
        "clf__shrinking": Categorical([True, False]),
    },
    "NuSVC": {
        "clf__nu": Real(0.1, 0.5, prior="uniform"),
        "clf__kernel": Categorical(["linear", "rbf", "poly", "sigmoid"]),
        "clf__gamma": Categorical(["scale", "auto"]),
        "clf__degree": Integer(2, 5),
        "clf__coef0": Real(0.0, 1.0),
    },
    "LogisticRegression": {
        "clf__C": Real(1e-3, 1e3, prior="log-uniform"),
        "clf__penalty": Categorical(["l2"]),
        "clf__solver": Categorical(["liblinear", "saga"]),
        "clf__tol": Real(1e-6, 1e-2, prior="log-uniform"),
    },
    "CalibratedRidge": {
        "clf__estimator__alpha": Real(0.1, 100.0, prior="log-uniform"),
        "clf__estimator__solver": Categorical(["auto", "sag", "saga", "lsqr"]),
    },
    "CalibratedLinearSVC": {
        "clf__estimator__C": Real(0.01, 10.0, prior="log-uniform"),
        "clf__estimator__loss": Categorical(["squared_hinge"]),
        "clf__estimator__max_iter": Integer(500, 3000),
    },
 
    "QDA": {
        "clf__reg_param": Real(0.0, 1.0, prior="uniform"),
    },
    "AdaBoost": {
        "clf__n_estimators": Integer(50, 200, name="n_estimators"),
        "clf__learning_rate": Real(0.01, 1.0, prior="log-uniform", name="learning_rate"),
        "clf__algorithm": Categorical(["SAMME"], name="algorithm"),
        "clf__random_state": Integer(1, 10, name="random_state"),
    },
 
}

# -------------------------------------------------------------------------
# 9. Bayesian hyperparameter optimization (inner CV)
# -------------------------------------------------------------------------

cv_inner = StratifiedKFold(
    n_splits=N_SPLITS_CV,
    shuffle=True,
    random_state=RANDOM_STATE,
)

best_estimators = {}

for model_name, base_model in models.items():
    print(f"\nTuning {model_name}...")

    model_pipeline = Pipeline([
        ("clf", base_model),
    ])

    if model_name in param_spaces:
        bayes_search = BayesSearchCV(
            estimator=model_pipeline,
            search_spaces=param_spaces[model_name],
            n_iter=50,
            scoring="balanced_accuracy",
            cv=cv_inner,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        bayes_search.fit(x_train_rfe, y_train_series)
        best_estimators[model_name] = bayes_search.best_estimator_
        print(f"Best Parameters: {bayes_search.best_params_}")
    else:
        model_pipeline.fit(x_train_rfe, y_train_series)
        best_estimators[model_name] = model_pipeline

# -------------------------------------------------------------------------
# 10. Cross-validated performance and learning curves
# -------------------------------------------------------------------------

cv_outer = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

for model_name, tuned_model in best_estimators.items():
    cv_scores = cross_val_score(
        tuned_model,
        x_train_rfe,
        y_train_series,
        cv=cv_outer,
        scoring="balanced_accuracy",
    )
    print(f"{model_name}: {cv_scores.mean():.4f} ({cv_scores.std():.4f})")

    plot_learning_curve(
        tuned_model,
        x_train_rfe,
        y_train_series,
        cv=cv_outer,
        model_name=model_name,
    )

# -------------------------------------------------------------------------
# 11. Final evaluation with leak-free threshold optimization
# -------------------------------------------------------------------------

results_summary = {}
results_doc = Document()
results_doc.add_heading(
    "Classification Report Summary (Leak-Free Threshold)",
    level=1,
)

for model_name, tuned_model in best_estimators.items():
    print(f"\nEvaluating {model_name}...")

    # Fit on the full training data
    tuned_model.fit(x_train_rfe, y_train_series)

    # Estimate optimal threshold using CV on training only
    optimal_threshold = estimate_threshold_via_cv(
        tuned_model,
        x_train_rfe,
        y_train_series.values.ravel(),
        cv_splits=5,
    )

    # Evaluation on held‑out test set
    y_test_pred = plot_model_evaluation(
        tuned_model,
        x_test_rfe,
        y_test,
        model_name=model_name,
        feature_names=selected_feature_names,
        threshold=optimal_threshold,
    )

    classification_stats = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
    )
    accuracy = accuracy_score(y_test, y_test_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    precision_macro = classification_stats["macro avg"]["precision"]
    recall_macro = classification_stats["macro avg"]["recall"]
    f1_macro = classification_stats["macro avg"]["f1-score"]

    results_summary[model_name] = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-Score (Macro)": f1_macro,
        "Best Threshold (train)": optimal_threshold,
    }

    print(
        f"{model_name} — Test Balanced Acc: {balanced_acc:.3f}, "
        f"Train-based Threshold: {optimal_threshold:.2f}"
    )

    # Add model results to Word report
    results_doc.add_heading(model_name, level=2)
    metric_table = results_doc.add_table(
        rows=1,
        cols=len(results_summary[model_name]) + 1,
    )
    metric_table.style = "Light Shading Accent 1"

    header_cells = metric_table.rows[0].cells
    header_cells[0].text = "Metric"
    for idx, metric_name in enumerate(results_summary[model_name].keys(), start=1):
        header_cells[idx].text = metric_name

    row_cells = metric_table.add_row().cells
    row_cells[0].text = model_name
    for idx, metric_value in enumerate(results_summary[model_name].values(), start=1):
        row_cells[idx].text = f"{metric_value:.4f}"

    # Attach evaluation figure
    eval_figure_path = f"{model_name}_evaluation.png"
    if os.path.exists(eval_figure_path):
        results_doc.add_picture(eval_figure_path, width=Inches(6))

# Save Word document with all model results
results_doc.save(WORD_RESULTS_PATH)
print(f"\nResults saved to '{WORD_RESULTS_PATH}'")

# Convert summary to DataFrame for quick comparison in console
results_df = pd.DataFrame(results_summary).T
results_df = results_df.sort_values(
    by="Balanced Accuracy",
    ascending=False,
)
print("\nClassification Report Summary (Leak-Free):")
print(results_df)
