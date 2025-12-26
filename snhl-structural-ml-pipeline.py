"""
Structural MRI ML pipeline for SNHL classification.

Main steps
----------
1. Load FreeSurfer-like .txt files and build a subject × feature table.
2. Clean and type‑cast features; create labels.
3. Save structural feature matrix to CSV.
4. Exploratory visualization (3D surface, correlation matrix).
5. Train/test split with stratification.
6. Outlier removal and correlation‑based feature pruning.
7. Scaling and RFE with LightGBM.
8. Bayesian hyperparameter tuning for selected models (NuSVC here).
9. Cross‑validated performance and learning curves.
10. Threshold optimization without data leakage.
11. Final test evaluation, plots, and Word report export.
"""

import os
import re
import glob
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

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

from sklearn.svm import NuSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC, LinearSVC
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
# Configuration
# -------------------------------------------------------------------------

INPUT_TABLE_FOLDER = r"D:/table"
STRUCTURAL_CSV_OUTPUT = r"D:\snhl\structural_output.csv"

CORR_STRUCT_ALL_PATH = r"D:\corr_structural_all.png"
CORR_STRUCT_SELECTED_PATH = r"D:\corr_structural_selected.png"
WORD_SELECTED_FEATURES_PATH = r"D:\table_structural_selected_features.docx"
WORD_RESULTS_PATH = r"D:\model_results_structural_LeakFree.docx"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 10
N_FEATURES_RFE = 8
CORRELATION_THRESHOLD = 0.9


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def extract_subject_data(txt_path):
    """
    Extract key–value pairs from a .txt file with tab-separated header and values.

    Assumes:
        line 1: header (field names)
        line 2: values
    Returns:
        dict mapping header -> value.
    """
    with open(txt_path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        try:
            header = next(reader)
            values = next(reader)
            return dict(zip(header, values))
        except StopIteration:
            print(f"Skipping file {txt_path}: less than 2 lines.")
            return {}


def convert_numeric_columns(data_frame):
    """
    Convert columns that are fully numeric (allowing a single decimal point) to float.

    Columns with any non-numeric values are left as object.
    """
    df = data_frame.copy()
    for col in df.columns:
        col_as_str = df[col].astype(str)
        is_numeric_like = col_as_str.str.replace(".", "", 1).str.isnumeric().all()
        if is_numeric_like:
            df[col] = pd.to_numeric(df[col])
    return df


def remove_extreme_outliers(
    data_frame,
    feature_threshold=0.7,
    subject_threshold=0.1,
):
    """
    Remove subjects with a high proportion of outlier features.

    For each numeric feature, outliers are defined using the 1.5 * IQR rule.
    A subject is flagged if (#outlier_features / total_features) > feature_threshold.
    Subjects are removed if the proportion of flagged subjects is < subject_threshold.
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


def plot_correlation_matrix(data_frame, title=None, save_path=None):
    """
    Plot a compact correlation heatmap for numeric columns.

    Removes constant features and rows with NaNs before computing correlations.
    """
    numeric_data = data_frame.select_dtypes(include=[np.number])
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
    Plot learning curve for an estimator using cross-validation and balanced accuracy.
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        x_data,
        y_data,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
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
    For probabilistic classifiers, find the decision threshold that maximizes
    balanced accuracy on a validation set. Returns 0.5 for non‑probabilistic models.
    """
    if not hasattr(estimator, "predict_proba"):
        return 0.5

    y_proba = estimator.predict_proba(x_val)[:, 1]
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5
    best_bal_acc = 0.0

    for threshold in candidate_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = threshold

    return best_threshold


def estimate_threshold_via_cv(estimator, x_data, y_data, cv_splits=5):
    """
    Estimate an optimal decision threshold using internal cross-validation
    on the training data only (leakage‑free).
    """
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    thresholds = []

    for train_idx, val_idx in cv.split(x_data, y_data):
        x_train_cv, x_val_cv = x_data[train_idx], x_data[val_idx]
        y_train_cv, y_val_cv = y_data[train_idx], y_data[val_idx]

        estimator.fit(x_train_cv, y_train_cv)
        threshold = find_best_threshold(estimator, x_val_cv, y_val_cv)
        thresholds.append(threshold)

    return float(np.mean(thresholds)) if thresholds else 0.5


def plot_model_evaluation(
    estimator,
    x_data,
    y_true,
    model_name,
    feature_names=None,
    threshold=0.5,
):
    """
    Plot confusion matrix, ROC, PR curves and feature importance (if available).

    Saves figure as '{model_name}_evaluation.png' and returns predicted labels.
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
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
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
        avg_prec = average_precision_score(y_true, y_proba)
        axes[2].plot(recall, precision, label=f"AP = {avg_prec:.2f}")
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
        indices = np.argsort(importances)[::-1]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        sorted_names = np.array(feature_names)[indices]
        axes[3].barh(sorted_names[::-1], importances[indices][::-1])
        axes[3].set_title("Feature Importance")
        axes[3].set_xlabel("Importance")
    else:
        axes[3].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{model_name}_evaluation.png", dpi=300)
    plt.show()

    return y_pred


def plot_3d_surface_from_dataframe(data_frame, title="3D Surface Plot of DataFrame"):
    """
    Plot 3D surface: X = feature index, Y = subject index, Z = feature value.
    """
    numeric_df = data_frame.select_dtypes(include=[np.number])
    x_axis = np.arange(numeric_df.shape[1])
    y_axis = np.arange(numeric_df.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    z_values = numeric_df.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(x_mesh, y_mesh, z_values, cmap="viridis")
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Feature Value")

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Subject Index")
    ax.set_zlabel("Feature Value")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Step 1–3: Build structural feature table and save CSV
# -------------------------------------------------------------------------

# Collect subject-wise data from all .txt files
subject_data = {}

for txt_file in glob.glob(os.path.join(INPUT_TABLE_FOLDER, "*.txt")):
    raw_id = os.path.basename(txt_file).split("_")[0]
    # Normalize subject ID: e.g. test-sub01 -> subject_01
    subject_id = re.sub(
        r"(test-)?(sub|subject)[-_]?(\d+)",
        r"subject_\3",
        raw_id,
    )

    if subject_id not in subject_data:
        subject_data[subject_id] = {}

    extracted = extract_subject_data(txt_file)
    subject_data[subject_id].update(extracted)

# Convert dict -> DataFrame (subjects in rows)
struct_df = pd.DataFrame.from_dict(subject_data, orient="index", dtype=str)

# Drop non-feature column if present
if "Measure:volume" in struct_df.columns:
    struct_df = struct_df.drop(columns="Measure:volume")

# Natural sort index and reset
struct_df = struct_df.reindex(natsorted(struct_df.index))
struct_df = struct_df.reset_index(drop=True)

# Create synthetic label (0: healthy, 1: patient)
struct_df["label"] = struct_df.index.to_series().apply(
    lambda idx: 0 if int(idx) < 30 else 1
)

# Clean column names for readability
struct_df.columns = struct_df.columns.str.replace("-", " ", regex=False)
struct_df.columns = struct_df.columns.str.replace("_", " ", regex=False)

# Convert numeric columns
struct_df = convert_numeric_columns(struct_df)

# Keep only numeric columns (including label)
struct_df = struct_df.select_dtypes(exclude="object")

# Persist structural matrix to CSV for reuse
struct_df.to_csv(STRUCTURAL_CSV_OUTPUT, index=False)

# -------------------------------------------------------------------------
# Step 4: 3D visualization
# -------------------------------------------------------------------------

plot_3d_surface_from_dataframe(struct_df, title="3D Surface Plot: Structural Features")

# -------------------------------------------------------------------------
# Step 5: Train/test split
# -------------------------------------------------------------------------

labels = struct_df["label"].values
feature_df = struct_df.drop(columns="label")

x_train, x_test, y_train, y_test = train_test_split(
    feature_df,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels,
)

x_train = pd.DataFrame(x_train, columns=feature_df.columns)
x_test = pd.DataFrame(x_test, columns=feature_df.columns)
y_train_series = pd.Series(y_train, name="label")

# -------------------------------------------------------------------------
# Step 6: Outlier removal and correlation-based feature pruning
# -------------------------------------------------------------------------

x_train_clean = remove_extreme_outliers(x_train)
x_test_clean = x_test[x_train_clean.columns]

corr_matrix = x_train_clean.corr()
label_corr = x_train_clean.corrwith(y_train_series).abs()

high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            drop_col = col_i if label_corr[col_i] < label_corr[col_j] else col_j
            high_corr_features.add(drop_col)

x_train_pruned = x_train_clean.drop(columns=high_corr_features)
x_test_pruned = x_test_clean.drop(columns=high_corr_features)

plot_correlation_matrix(
    feature_df,
    title="Correlation Matrix: Structural Features (All)",
    save_path=CORR_STRUCT_ALL_PATH,
)

# -------------------------------------------------------------------------
# Step 7: Scaling and RFE with LightGBM
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

selected_train_df = pd.DataFrame(x_train_rfe, columns=selected_feature_names)
plot_correlation_matrix(
    selected_train_df,
    title="Correlation Matrix: Selected Structural Features",
    save_path=CORR_STRUCT_SELECTED_PATH,
)

# Optionally save selected feature names into a Word table
selected_feature_doc = Document()
selected_feature_doc.add_heading("Selected Structural Features", level=1)
feature_table = selected_feature_doc.add_table(
    rows=len(selected_feature_names),
    cols=1,
)
for row_idx, feature_name in enumerate(selected_feature_names):
    feature_table.cell(row_idx, 0).text = feature_name
selected_feature_doc.save(WORD_SELECTED_FEATURES_PATH)

# -------------------------------------------------------------------------
# Step 8: Model definitions (using NuSVC here; others kept in param grid)
# -------------------------------------------------------------------------

models = {
       'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    # # #
    
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'NuSVC': NuSVC(probability=True, random_state=42,class_weight='balanced' ),
    'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
    
    
    'CalibratedRidge': CalibratedClassifierCV(
        estimator=RidgeClassifier(random_state=42, class_weight = 'balanced', max_iter = 1000),method='isotonic', cv=10
    ),
    'CalibratedLinearSVC': CalibratedClassifierCV(
        estimator=LinearSVC(random_state=42, dual=False, class_weight = 'balanced', max_iter = 1000), method='isotonic', cv=10
    ),
    

    'QDA': QuadraticDiscriminantAnalysis(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
}

param_spaces = {
    # Kept full search spaces for reproducibility and potential future use
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
        "clf__n_estimators": Integer(200, 700),
        "clf__learning_rate": Real(0.02, 0.15, prior="log-uniform"),
        "clf__max_depth": Integer(4, 8),
        "clf__subsample": Real(0.7, 1.0),
        "clf__colsample_bytree": Real(0.7, 1.0),
        "clf__reg_alpha": Real(0.1, 5.0, prior="log-uniform"),
        "clf__reg_lambda": Real(0.5, 10.0, prior="log-uniform"),
    },
    "CatBoost": {
        "clf__iterations": Integer(50, 300),
        "clf__depth": Integer(3, 10),
        "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "clf__l2_leaf_reg": Real(1.0, 10.0),
        "clf__border_count": Integer(32, 255),
    },
    "KNN": {
        "clf__n_neighbors": Integer(15, 80),
        "clf__weights": Categorical(["distance"]),
        "clf__p": Integer(1, 2),
        "clf__algorithm": Categorical(["auto", "ball_tree", "kd_tree"]),
        "clf__leaf_size": Integer(20, 60),
        "clf__metric": Categorical(["minkowski"]),
    },
    "SVM": {
        "clf__C": Real(1e-3, 1.0, prior="log-uniform"),
        "clf__kernel": Categorical(["linear", "rbf", "poly"]),
        "clf__gamma": Real(1e-5, 1e-2, prior="log-uniform"),
        "clf__degree": Categorical([2]),
        "clf__shrinking": Categorical([True]),
        "clf__class_weight": Categorical([None, "balanced"]),
    },
    "NuSVC": {
        "clf__nu": Real(0.03, 0.10, prior="uniform"),
        "clf__kernel": Categorical(["linear", "rbf"]),
        "clf__gamma": Real(1e-6, 5e-4, prior="log-uniform"),
        "clf__class_weight": Categorical(["balanced"]),
        "clf__shrinking": Categorical([True]),
    },
    "LogisticRegression": {
        "clf__C": Real(1e-3, 1e3, prior="log-uniform"),
        "clf__penalty": Categorical(["l2"]),
        "clf__solver": Categorical(
            ["liblinear", "saga", "lbfgs", "newton-cg", "sag"]
        ),
        "clf__tol": Real(1e-6, 1e-2, prior="log-uniform"),
        "clf__fit_intercept": Categorical([True, False]),
        "clf__class_weight": Categorical(["balanced", None]),
        "clf__l1_ratio": Real(0.0, 1.0),
        "clf__max_iter": Categorical([100, 200, 300, 500, 1000]),
        "clf__warm_start": Categorical([True, False]),
        "clf__dual": Categorical([False]),
    },
    "RidgeClassifier": {
        "clf__alpha": Real(0.1, 100.0, prior="log-uniform"),
        "clf__solver": Categorical(["auto", "sag", "saga", "lsqr"]),
    },
    "CalibratedRidge": {
        "clf__estimator__alpha": Real(1e-4, 1e2, prior="log-uniform"),
        "clf__estimator__solver": Categorical(
            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        ),
        "clf__estimator__tol": Real(1e-5, 1e-2, prior="log-uniform"),
        "clf__estimator__fit_intercept": Categorical([True, False]),
        "clf__estimator__max_iter": Integer(1000, 10000),
    },
    "CalibratedLinearSVC": {
        "clf__estimator__C": Real(1e-3, 1e2, prior="log-uniform"),
        "clf__estimator__tol": Real(1e-5, 1e-1, prior="log-uniform"),
        "clf__estimator__loss": Categorical(["squared_hinge"]),
        "clf__estimator__class_weight": Categorical([None, "balanced"]),
        "clf__estimator__max_iter": Integer(1000, 10000),
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
# Step 9: Bayesian hyperparameter tuning (NuSVC)
# -------------------------------------------------------------------------

best_estimators = {}
cv_inner = StratifiedKFold(
    n_splits=N_SPLITS_CV,
    shuffle=True,
    random_state=RANDOM_STATE,
)

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
# Step 10: Cross-validated performance and learning curves
# -------------------------------------------------------------------------

results_cv = []
model_names_cv = []

for model_name, tuned_model in best_estimators.items():
    cv_outer = StratifiedKFold(
        n_splits=10,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    cv_scores = cross_val_score(
        tuned_model,
        x_train_rfe,
        y_train_series,
        cv=cv_outer,
        scoring="balanced_accuracy",
    )
    results_cv.append(cv_scores)
    model_names_cv.append(model_name)
    print(f"{model_name}: {cv_scores.mean():.4f} ({cv_scores.std():.4f})")

    plot_learning_curve(
        tuned_model,
        x_train_rfe,
        y_train_series,
        cv=cv_outer,
        model_name=model_name,
    )

# -------------------------------------------------------------------------
# Step 11: Final evaluation with leak-free threshold optimization
# -------------------------------------------------------------------------

results_summary = {}
results_doc = Document()
results_doc.add_heading(
    "Classification Report Summary (Leak-Free Threshold)",
    level=1,
)

for model_name, tuned_model in best_estimators.items():
    print(f"\nEvaluating {model_name}...")

    tuned_model.fit(x_train_rfe, y_train_series)

    best_threshold = estimate_threshold_via_cv(
        tuned_model,
        x_train_rfe,
        y_train_series.values.ravel(),
        cv_splits=5,
    )

    y_test_pred = plot_model_evaluation(
        tuned_model,
        x_test_rfe,
        y_test,
        model_name=model_name,
        feature_names=selected_feature_names,
        threshold=best_threshold,
    )

    report_dict = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
    )
    accuracy = accuracy_score(y_test, y_test_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    precision_macro = report_dict["macro avg"]["precision"]
    recall_macro = report_dict["macro avg"]["recall"]
    f1_macro = report_dict["macro avg"]["f1-score"]

    results_summary[model_name] = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-Score (Macro)": f1_macro,
        "Best Threshold (train)": best_threshold,
    }

    print(
        f"{model_name} — Test Balanced Acc: {balanced_acc:.3f}, "
        f"Train-based Threshold: {best_threshold:.2f}"
    )

    # Add to Word document
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

    eval_fig_path = f"{model_name}_evaluation.png"
    if os.path.exists(eval_fig_path):
        results_doc.add_picture(eval_fig_path, width=Inches(6))

results_doc.save(WORD_RESULTS_PATH)
print(f"\nResults saved to '{WORD_RESULTS_PATH}'")

results_df = pd.DataFrame(results_summary).T
results_df = results_df.sort_values(
    by="Balanced Accuracy",
    ascending=False,
)
print("\nClassification Report Summary (Leak-Free):")
print(results_df)


