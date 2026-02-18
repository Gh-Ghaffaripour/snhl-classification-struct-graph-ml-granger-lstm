"""
 Whole MRI ML pipeline

- Reads precomputed whole-brain CSV feature tables
- Merges features into a subject-level DataFrame
- Cleans and preprocesses data
- Removes outliers and highly correlated features
- Performs feature selection via RFE with LightGBM
- Tunes multiple classifiers using Bayesian optimization
- Evaluates models with leak-free threshold selection
- Generates figures and exports a Word report

Author: <Hossein Gharedaghi>
"""


# Imports


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)

from sklearn.impute import KNNImputer
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

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from scipy.stats import ttest_ind
from docx import Document
from docx.shared import Inches



# Configuration


WHOLE_CSV      = r"Whole_output.csv"
OUTPUT_DIR     = r"outputs"

CORR_FIG_1 = os.path.join(OUTPUT_DIR, "corr_Functional_1.png")
CORR_FIG_2 = os.path.join(OUTPUT_DIR, "corr_Functional_2.png")
CORR_FIG_SELECTED = os.path.join(OUTPUT_DIR, "corr_Functional_selected.png")
DOC_MODEL_RESULTS = os.path.join(
    OUTPUT_DIR, "model_results_Functional_LeakFree.docx"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Utility Functions


def remove_extreme_outliers(
    df: pd.DataFrame,
    feature_threshold: float = 0.7,
    subject_threshold: float = 0.1,
) -> pd.DataFrame:
    """Remove subjects with excessive outlier features (IQR-based)."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    total_features = len(numeric_cols)

    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_mask[col] = (df[col] < lb) | (df[col] > ub)

    df = df.copy()
    df["outlier_frac"] = outlier_mask.sum(axis=1) / total_features

    if (df["outlier_frac"] > feature_threshold).mean() < subject_threshold:
        df = df[df["outlier_frac"] <= feature_threshold]

    return df.drop(columns="outlier_frac")


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = None,
    save_path: str = None,
) -> None:
    """Plot correlation heatmap for numeric features."""
    data = data.select_dtypes(include=[np.number])
    data = data.loc[:, data.nunique() > 1].dropna(axis=0)

    corr = data.corr()
    if corr.empty:
        return

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        corr,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6},
    )

    if title:
        plt.title(title, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

# Threshold Optimization (Leak-Free)


def find_best_threshold(model, X_val, y_val):
    if not hasattr(model, "predict_proba"):
        return 0.5

    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)

    best_t, best_score = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(y_val, preds)
        if score > best_score:
            best_score, best_t = score, t

    return best_t


def estimate_threshold_cv(model, X, y, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    thresholds = []

    for tr, va in cv.split(X, y):
        model.fit(X[tr], y[tr])
        thresholds.append(find_best_threshold(model, X[va], y[va]))

    return float(np.mean(thresholds))

def welch_t_filter(X_train: pd.DataFrame,
                   y_train: np.ndarray,
                   alpha: float = 0.01) -> pd.DataFrame:
    """
    Feature selection with Welch's t-test on training data.
    Keeps features with p < alpha (uncorrected).
    """
    selected_cols = []
    for col in X_train.columns:
        group0 = X_train[y_train == 0][col]
        group1 = X_train[y_train == 1][col]
        stat, p = ttest_ind(group0, group1, equal_var=False)
        if p < alpha:
            selected_cols.append(col)
    return X_train[selected_cols], selected_cols

# Main Pipeline



def main():


    # Load & merge data
  
    df = pd.read_csv(WHOLE_CSV)


    df = df.drop(columns=["0 Subject/Sample #", "label"], errors="ignore")

    # Label definition
    df["label"] = df.index.to_series().astype(int).apply(
        lambda x: 0 if x < 30 else 1
    )



    X = df.drop("label", axis=1)
    y = df["label"].values
#
#

     # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
#

    # Imputation

    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)
#

#     # Outlier removal (exact same as first script)

    X_train = remove_extreme_outliers(X_train)
    X_test = X_test[X_train.columns]

    plot_correlation_matrix(
        X_train, "Functional pipeline", CORR_FIG_1
    )


#     # Correlation filtering (mimic first script exactly)

    y_train_df = pd.DataFrame(y_train, columns=["label"])

    corr_orig = X_train.corr()
    threshold = 0.9


    label_corr = X_train.corrwith(y_train_df).abs()

    highly_correlated = set()
    for i in range(len(corr_orig.columns)):
        for j in range(i):
            if abs(corr_orig.iloc[i, j]) > threshold:
                f_i = corr_orig.columns[i]
                f_j = corr_orig.columns[j]
                drop = f_i if label_corr[f_i] < label_corr[f_j] else f_j
                highly_correlated.add(drop)

    X_train = X_train.drop(columns=highly_correlated)
    X_test = X_test[X_train.columns]


#
    plot_correlation_matrix(
        X_train, "Combined (reduced)", CORR_FIG_2
    )

    X_train, t_selected = welch_t_filter(X_train, y_train, alpha=0.01)
    X_test = X_test[X_train.columns]



     # Scaling

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)


    # RFE with LightGBM

    lgbm = LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )

    rfe = RFE(lgbm, n_features_to_select=6,step=1)
    rfe.fit(X_train_s, y_train)

    X_train_rfe = rfe.transform(X_train_s)
    X_test_rfe = rfe.transform(X_test_s)

    selected_features = X_train.columns[rfe.get_support()].tolist()
    print("Selected features:", selected_features)


    plot_correlation_matrix(
        pd.DataFrame(X_train_rfe, columns=selected_features),
        "Selected combined features",
        CORR_FIG_SELECTED,
    )
#
#     # Models & Bayesian optimization

    models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    #

    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'NuSVC': NuSVC(probability=True, random_state=42,class_weight="balanced"),
    'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),


    'CalibratedRidge': CalibratedClassifierCV(
        estimator=RidgeClassifier(random_state=42), method='sigmoid', cv=5
    ),
    'CalibratedLinearSVC': CalibratedClassifierCV(
        estimator=LinearSVC(random_state=42, dual=False), method='sigmoid', cv=5
    ),
#

    'QDA': QuadraticDiscriminantAnalysis(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
# #     # #

}


# Hyperparameter spaces

    search_spaces = {
        'RandomForest': {
        'clf__n_estimators': Integer(50, 300),
        'clf__max_depth': Integer(3, 20),
        'clf__min_samples_split': Integer(2, 20),
        'clf__min_samples_leaf': Integer(1, 10),
        'clf__max_features': Categorical(['sqrt', 'log2', None]),
        'clf__bootstrap': Categorical([True, False]),
        'clf__criterion': Categorical(['gini', 'entropy', 'log_loss'])
    },
    #
    'ExtraTrees': {
    'clf__n_estimators': Integer(80, 300), 
    'clf__max_depth': Integer(4, 12), 
    'clf__min_samples_split': Integer(15, 40),  
    'clf__min_samples_leaf': Integer(10, 30),  
    'clf__max_features': Categorical(['sqrt']),  
    'clf__bootstrap': Categorical([True]),
    'clf__max_samples': Real(0.5, 0.8),  
    'clf__min_impurity_decrease': Real(0.001, 0.02),  
    'clf__ccp_alpha': Real(0.0, 0.01)  
},



        'GradientBoosting': {
    'clf__n_estimators': Integer(50, 250), 
    'clf__learning_rate': Real(0.01, 0.15, prior='log-uniform'),  
    'clf__max_depth': Integer(3, 6),
    'clf__subsample': Real(0.6, 0.9),  
    'clf__loss': Categorical(['log_loss']),
    'clf__min_samples_split': Integer(10, 30),  
    'clf__min_samples_leaf': Integer(15, 80), 
    'clf__max_features': Categorical(['sqrt', 'log2']),
    'clf__validation_fraction': Real(0.1, 0.2),  
    'clf__n_iter_no_change': Integer(10, 20) ,
},


    'HistGradientBoosting': {
        'clf__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'clf__max_iter': Integer(100, 500),
        'clf__max_depth': Integer(3, 12),
        'clf__min_samples_leaf': Integer(10, 50),
        'clf__l2_regularization': Real(0.0, 1.0),
        'clf__max_bins': Integer(50, 255)
    },

    
         'XGBoost': {
    'clf__n_estimators': Integer(150, 500),
    'clf__learning_rate': Real(0.01, 0.12, prior='log-uniform'),  
    'clf__max_depth': Integer(3, 6), 
    'clf__min_child_weight': Integer(3, 10), 
    'clf__subsample': Real(0.6, 0.85),  
    'clf__colsample_bytree': Real(0.6, 0.85),  
    'clf__colsample_bylevel': Real(0.6, 0.9),  
    'clf__reg_alpha': Real(0.5, 8.0, prior='log-uniform'),
    'clf__reg_lambda': Real(1.0, 15.0, prior='log-uniform'),  
    'clf__gamma': Real(0.1, 1.0, prior='log-uniform') 
},


    #
    'KNN': {
    'clf__n_neighbors': Integer(5, 25), 
    'clf__weights': Categorical(['uniform', 'distance']),
    'clf__p': Categorical([2]), 
    'clf__algorithm': Categorical(['auto']), 
    'clf__leaf_size': Integer(20, 50)  
},
#
   #
     'SVM': {
    'clf__C': Real(0.001, 0.5, prior='log-uniform'),  
    'clf__kernel': Categorical(['linear']), 
    'clf__gamma': Real(1e-5, 5e-3, prior='log-uniform'),  
    'clf__degree': Categorical([2]),
    'clf__shrinking': Categorical([True]),
    'clf__class_weight': Categorical(['balanced', None]),
    'clf__tol': Real(1e-3, 1e-2, prior='log-uniform'),  
    'clf__max_iter': Integer(1000, 5000) 
},



   'NuSVC': {
    'clf__nu': Real(0.1, 0.5, prior='uniform'),
    'clf__kernel': Categorical(['linear', 'rbf', 'poly', 'sigmoid']),
    'clf__gamma': Categorical(['scale', 'auto']),
    'clf__degree': Integer(2, 5),
    'clf__coef0': Real(0.0, 1.0)
   },

             'LogisticRegression': {
    'clf__C': Real(0.00001, 0.01, prior='log-uniform', name='C'), 
    'clf__penalty': Categorical(['l1'], name='penalty'),  
    'clf__solver': Categorical(['saga'], name='solver'),  
    'clf__tol': Real(0.1, 0.5, prior='log-uniform', name='tol'),  
    'clf__fit_intercept': Categorical([False], name='fit_intercept'),  
    'clf__class_weight': Categorical([None], name='class_weight'), 
    'clf__max_iter': Categorical([20, 50], name='max_iter'),
    'clf__dual': Categorical([False], name='dual')
},

  
    'CalibratedRidge': {
        'clf__estimator__alpha': Real(0.1, 100.0, prior='log-uniform'),
        'clf__estimator__solver': Categorical(['auto', 'sag', 'saga', 'lsqr']),
    },
   'CalibratedLinearSVC': {
    'clf__estimator__C': Real(0.01, 10.0, prior='log-uniform'),
    'clf__estimator__loss': Categorical(['squared_hinge']),  # hinge removed
    'clf__estimator__max_iter': Integer(500, 3000),
    },

 
    'QDA': {
        'clf__reg_param': Real(0.0, 1.0, prior='uniform')
    },

      'AdaBoost': {
    'clf__n_estimators': Integer(50, 200, name='n_estimators'),  
    'clf__learning_rate': Real(0.01, 1.0, prior='log-uniform', name='learning_rate'),  
    'clf__algorithm': Categorical(['SAMME'], name='algorithm'), 
    'clf__random_state': Integer(1, 10, name='random_state') 
     },

}
#
    best_estimators = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in models.items():
        pipe = Pipeline([("clf", model)])
        search = BayesSearchCV(
            pipe,
            search_spaces[name],
            n_iter=50,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train_rfe, y_train)
        best_estimators[name] = search.best_estimator_


    # Leak-free test evaluation + Word report

    doc = Document()
    doc.add_heading(
        "Whole-Brain MRI Classification Results (Leak-Free)", level=1
    )

    for name, model in best_estimators.items():
        model.fit(X_train_rfe, y_train)
        thresh = estimate_threshold_cv(
            model, X_train_rfe, y_train
        )

        probs = model.predict_proba(X_test_rfe)[:, 1]
        preds = (probs >= thresh).astype(int)

        report = classification_report(
            y_test, preds, output_dict=True
        )

        doc.add_heading(name, level=2)
        table = doc.add_table(rows=1, cols=6)
        headers = [
            "Accuracy",
            "Balanced Acc",
            "Precision",
            "Recall",
            "F1",
            "Threshold",
        ]
        for i, h in enumerate(headers):
            table.rows[0].cells[i].text = h

        row = table.add_row().cells
        row[0].text = f"{accuracy_score(y_test, preds):.3f}"
        row[1].text = f"{balanced_accuracy_score(y_test, preds):.3f}"
        row[2].text = f"{report['macro avg']['precision']:.3f}"
        row[3].text = f"{report['macro avg']['recall']:.3f}"
        row[4].text = f"{report['macro avg']['f1-score']:.3f}"
        row[5].text = f"{thresh:.2f}"

    doc.save(DOC_MODEL_RESULTS)
    print(f"Results saved to: {DOC_MODEL_RESULTS}")


if __name__ == "__main__":
    main()
