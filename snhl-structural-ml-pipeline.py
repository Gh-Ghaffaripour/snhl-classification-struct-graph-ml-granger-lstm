import pandas as pd
from natsort import natsorted

# Initialize an empty dictionary to hold subject data
data_dict = {}

import csv
import re
import os
import glob
input_folder = 'D:/table'

def extract_subject_data(file_path):
    """Extracts data from a CSV file using tab as a delimiter."""
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter='\t') # Use tab delimiter for flexibility
        try:
            header = next(reader) # Read first line as header
            values = next(reader) # Read second line as values
            return dict(zip(header, values))
        except StopIteration:
            print(f"⚠️ Skipping file {file_path}: Unexpected format (less than 2 lines)")
            return {}

list1 = []
# Loop through all files in the input folder
for file in glob.glob(os.path.join(input_folder, "*.txt")):
        raw_id = os.path.basename(file).split("_")[0]  # Extract subject ID
        subject_id = re.sub(r'(test-)?(sub|subject)[-_]?(\d+)', r'subject_\3', raw_id)

        if subject_id not in data_dict:
           data_dict[subject_id] = {}
        extract_data = extract_subject_data(file)
        data_dict[subject_id].update(extract_data)
        # for i in subject_id:
        #     i.split("-")[1]
        #     print(i)

data_dict
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=str)
df
df = df.drop('Measure:volume', axis=1)
df = df.reindex(natsorted(df.index))
df
df = df.reset_index()
# df
df['label'] = df.index.to_series().apply(lambda x: 0 if int(x) < 30 else 1) # label 0 => healthy
# df
df.columns =  df.columns.str.replace('-'," ")
df.columns = df.columns.str.replace('_'," ")
pd.isna(df).sum()
df.duplicated().sum()
df
def convert_numeric_columns(df):
    for col in df.columns:
        if df[col].astype(str).str.replace('.','',1).str.isnumeric().all():
            df[col] = pd.to_numeric(df[col])
    return df
df = convert_numeric_columns(df)
df.dtypes
df = df.select_dtypes(exclude='object')
df.to_csv(r'D:\snhl\structural_output.csv', index=False)
df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Create meshgrid for X and Y
x = np.arange(df.select_dtypes(include=[np.number]).shape[1]) # feature index
y = np.arange(df.select_dtypes(include=[np.number]).shape[0]) # subject index
x, y = np.meshgrid(x, y)

# Prepare Z values from the DataFrame
z = df.select_dtypes(include=[np.number]).values

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='viridis')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Feature Value')

# Labels
ax.set_xlabel('Feature Index')
ax.set_ylabel('Subject Index')
ax.set_zlabel('Feature Value')
plt.title('3D Surface Plot of DataFrame')
plt.show()
Y = df['label'].values
df_in = df.drop('label',axis=1)
X = df_in
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 1, stratify=Y)

X_train = pd.DataFrame(X_train, columns=X_train.columns)
X_test = pd.DataFrame(X_test, columns=X_train.columns)
y_train = pd.DataFrame(y_train, columns=["label"])
y_train

def remove_extreme_outliers(df, feature_threshold=0.7, subject_threshold=0.1):
    numeric_cols = df.select_dtypes(include=['number']).columns
    total_features = len(numeric_cols)
    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)


    df['outlier_count'] = outlier_mask.sum(axis=1)

    df['outlier_subject'] = (df['outlier_count'] / total_features) > feature_threshold

    num_outlier_subjects = df['outlier_subject'].sum()
    total_subjects = len(df)


    if (num_outlier_subjects / total_subjects) < subject_threshold:
        df = df[~df['outlier_subject']]


    df = df.drop(columns=['outlier_count', 'outlier_subject'])


    return df

X_train = remove_extreme_outliers(X_train)
X_test = X_test[X_test.columns]
import matplotlib.pyplot as plt
import seaborn as sns
def plot_correlation_matrix(data, title= None, save_path=None):
    # Step 1: Keep only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Step 2: Remove constant features
    numeric_data = numeric_data.loc[:, numeric_data.nunique() > 1]

    # Step 3: Drop any rows with NaNs
    numeric_data = numeric_data.dropna(axis=0)

    # Step 4: Compute correlation matrix
    corr_matrix = numeric_data.corr()

    # Step 5: Safety check
    if corr_matrix.empty or corr_matrix.shape[0] != corr_matrix.shape[1]:
        print("Correlation matrix is empty or invalid.")
        return

    # Step 6: Create upper triangle mask of the same shape
    mask = np.triu(np.ones(corr_matrix.shape), k=0).astype(bool)

    # Sanity check (optional debug print)
    assert mask.shape == corr_matrix.shape, "Mask shape does not match correlation matrix shape"

    # Step 7: Plot
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        corr_matrix,
        # mask=mask,
        cmap='viridis',
        center=0,
        vmin=-1,
        vmax=1,
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,

        linecolor='white',
        fmt=".2f",
        # annot=True,
        annot_kws={'size':0.4},
        cbar_kws={'shrink':0.6,'ticks': [-1,-0.5,0,0.5,1]}
    )
    plt.title(title, fontsize=10)
    # plt.xlabel(fontsize=4)
    # plt.ylabel(fontsize=4)
    plt.style.use('default')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
#
# plot_correlation_matrix(X_train, save_path= r"D:\corr_structural_1.png", title='Structural pipeline')

corr_orig = X_train.corr()
threshold = 0.9
label_corr = X_train.corrwith(y_train).abs()
highly_correlated = set()
for i in range(len(corr_orig.columns)):
    for j in range(i):
        if abs(corr_orig.iloc[i, j]) > threshold:
            f_i = corr_orig.columns[i]
            f_j = corr_orig.columns[j]
            drop = f_i if label_corr[f_i] < label_corr[f_j] else f_j
            highly_correlated.add(drop)
X_train = X_train.drop(columns=highly_correlated)
X_test = X_test.drop(columns=highly_correlated)

plot_correlation_matrix(X, save_path= r"D:\corr_structural_2.png")

X_train
from sklearn.preprocessing import StandardScaler
# 5. Scaling (fit only on training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, RFE
n_features_to_select = 8  # Enforce exactly 14 features

lgbm = LGBMClassifier(
    n_estimators=200,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42)

rfe = RFE(estimator=lgbm, n_features_to_select=n_features_to_select,step=1)

rfe.fit(X_train_scaled, y_train)
# 3. Transform Data with Selected Features
X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)

print('Train RFE Shape:', X_train_rfe.shape)
print('Test RFE Shape:', X_test_rfe.shape)
# rfe.get_support()
# Get selected feature names
selected_features = list(X_train.columns[rfe.get_support()])
print("Selected Features:", selected_features)
# X_train

print('Train RFE Shape:', X_train_rfe.shape)
print('Test RFE Shape:', X_test_rfe.shape)
# rfe.get_support()
# Get selected feature names
selected_features = list(X_train.columns[rfe.get_support()])
print("Selected Features:", selected_features)
df_train = pd.DataFrame(X_train_rfe, columns=selected_features)
plot_correlation_matrix(df_train, title="selected Structrural feature", save_path= r"D:\corr_structural_selected.png")
from docx import Document
selected_feature= [['Right Hippocampus'],
                   ['MaskVol to eTIV'],
                   [ 'lh cuneus thickness'],
                   ['rh isthmuscingulate curvind'],
                   ['rh superiorfrontal curvind'],
                   ['rh isthmuscingulate meancurv'],
                   ['rh posteriorcingulate meancurv'],
                   ['rh superiorfrontal thickness'],
                   ['rh caudalanteriorcingulate volume']]

# Create a new Word document
doc = Document()

# Add a table with rows and columns equal to the list size
table = doc.add_table(rows=len(selected_feature), cols=len(selected_feature[0]))

# Fill the table with data
for i, row in enumerate(selected_feature):
    for j, val in enumerate(row):
        table.cell(i, j).text = str(val)

# Save the document
doc.save(r"D:\\table__structural.docx")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skopt.space import Integer, Real, Categorical

# ----------------------
# Model Definitions
# ----------------------
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from skopt.space import Integer, Real, Categorical
from skopt.space import Real, Categorical
# ----------------------
# Models
# ----------------------
models = {
    # 'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    # 'ExtraTrees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
    # 'GradientBoosting': GradientBoostingClassifier(random_state=42),
    # 'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    # # # #
    #
    # 'KNN': KNeighborsClassifier(),
    # 'SVM': SVC(probability=True, random_state=42),
    'NuSVC': NuSVC(probability=True, random_state=42,class_weight='balanced' ),
    # 'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
    #
    #
    # 'CalibratedRidge': CalibratedClassifierCV(
    #     estimator=RidgeClassifier(random_state=42, class_weight = 'balanced', max_iter = 1000),method='isotonic', cv=10
    # ),
    # 'CalibratedLinearSVC': CalibratedClassifierCV(
    #     estimator=LinearSVC(random_state=42, dual=False, class_weight = 'balanced', max_iter = 1000), method='isotonic', cv=10
    # ),
    #

    # 'QDA': QuadraticDiscriminantAnalysis(),
    # 'AdaBoost': AdaBoostClassifier(random_state=42),
#     # #

}

# ----------------------
# Hyperparameter spaces
# ----------------------
params = {
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
        'clf__n_estimators': Integer(50, 300),
        'clf__max_depth': Integer(3, 20),
        'clf__min_samples_split': Integer(2, 20),
        'clf__min_samples_leaf': Integer(1, 10),
        'clf__max_features': Categorical(['sqrt', 'log2', None]),
    #     'clf__criterion': Categorical(['gini', 'entropy', 'log_loss'])
    },



    'GradientBoosting':  {
    'clf__n_estimators': Integer(200, 1200),        # allow larger with early stopping
    'clf__learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'clf__max_depth': Integer(3, 6),                # shallower trees
    'clf__subsample': Real(0.6, 0.9),               # add stochasticity
    'clf__loss': Categorical(['log_loss']),         # keep stable loss
    'clf__min_samples_split': Integer(10, 100),     # avoid tiny splits
    'clf__min_samples_leaf': Integer(20, 200),      # larger leaves → smoother
    # instead of 'sqrt'/'log2', try fraction of features
    'clf__max_features': Real(0.2, 0.8),
    # optional: limit number of leaves (alternative to max_depth)
    'clf__max_leaf_nodes': Integer(10, 200),
   },


    # 'GradientBoosting': {
    # 'clf__n_estimators': Integer(150, 400),                       # moderate boosting rounds
    # 'clf__learning_rate': Real(0.05, 0.2, prior='log-uniform'),  # slower, smoother learning
    # 'clf__max_depth': Integer(5, 9),                              # shallow trees → less overfit
    # 'clf__subsample': Real(0.8, 1.0),                             # stochastic boosting to improve generalization
    # 'clf__loss': Categorical(['log_loss']),                       # more stable than 'exponential'
    # 'clf__min_samples_split': Integer(5, 20),                     # prevent splitting on tiny noisy samples
    # 'clf__min_samples_leaf': Integer(10, 60),                     # larger leaves → smoother decisions
    # 'clf__max_features': Categorical(['sqrt', 'log2']),
    # },# subset of features for decorrelation


    # 'HistGradientBoosting': {
    #     'clf__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    #     'clf__max_iter': Integer(100, 500),
    #     'clf__max_depth': Integer(3, 12),
    #     'clf__min_samples_leaf': Integer(10, 50),
    #     'clf__l2_regularization': Real(0.0, 1.0),
    #     'clf__max_bins': Integer(50, 255)
    # },

    'HistGradientBoosting': {
        'clf__learning_rate': Real(0.05, 0.2, prior='log-uniform'),
        'clf__max_iter': Integer(30, 700),
        'clf__max_depth': Integer(5, 9),
        'clf__min_samples_leaf': Integer(10, 60),
        'clf__l2_regularization': Real(0.05, 1.5),
        'clf__max_bins': Integer(128, 255),
        'clf__early_stopping': Categorical([True]),
    },

    # 'XGBoost': {
    #     'clf__n_estimators': Integer(50, 300),
    #     'clf__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    #     'clf__max_depth': Integer(3, 12),
    #     'clf__subsample': Real(0.6, 1.0),
    #     'clf__colsample_bytree': Real(0.6, 1.0),
    #     'clf__reg_alpha': Real(1e-6, 10.0, prior='log-uniform'),
    #     'clf__reg_lambda': Real(1e-6, 10.0, prior='log-uniform')
    # },
     'XGBoost': {
    'clf__n_estimators': Integer(200, 700),                      # enough boosting rounds
    'clf__learning_rate': Real(0.02, 0.15, prior='log-uniform'), # lower → smoother learning
    'clf__max_depth': Integer(4, 8),                             # moderate depth
    'clf__subsample': Real(0.7, 1.0),                            # random sampling for regularization
    'clf__colsample_bytree': Real(0.7, 1.0),                     # column sampling (decorrelates trees)
    'clf__reg_alpha': Real(0.1, 5.0, prior='log-uniform'),        # L1 regularization
    'clf__reg_lambda': Real(0.5, 10.0, prior='log-uniform'),      # L2 regularization
    # 'clf__min_child_weight': Integer(3, 10),                      # prevents small leaves
    # 'clf__gamma': Real(0.0, 5.0),                                 # minimum loss reduction
    # 'clf__tree_method': Categorical(['hist']),
    },# efficient and consistent with HGB




    'CatBoost': {
        'clf__iterations': Integer(50, 300),
        'clf__depth': Integer(3, 10),
        'clf__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'clf__l2_leaf_reg': Real(1.0, 10.0),
        'clf__border_count': Integer(32, 255)
    },
   #
    'KNN': {
    'clf__n_neighbors': Integer(15, 80), # smoother, more generalizable
    'clf__weights': Categorical(['distance']), # smoother decision boundary
    'clf__p': Integer(1, 2),
    'clf__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree']),
    'clf__leaf_size': Integer(20, 60),
    'clf__metric': Categorical(['minkowski']),
},
   #

     'SVM': {
    # Moderate-to-strong regularization: smaller C → less overfit
    'clf__C': Real(1e-3, 1.0, prior='log-uniform'),

    # Use only robust kernels
    'clf__kernel': Categorical(['linear', 'rbf', 'poly']),

    # Smoother gamma for RBF/poly kernels
    'clf__gamma': Real(1e-5, 1e-2, prior='log-uniform'),

    # Keep polynomial degree small to avoid high variance
    'clf__degree': Categorical([2]),

    # Shrinking can stabilize convergence
    'clf__shrinking': Categorical([True]),

    # Enable class balancing if imbalance exists
    'clf__class_weight': Categorical([None, 'balanced']),
},



    'NuSVC':   {
    'clf__nu': Real(0.03, 0.10, prior='uniform'), # tighter margin control
    'clf__kernel': Categorical(['linear', 'rbf']),
    'clf__gamma': Real(1e-6, 5e-4, prior='log-uniform'), # smoother RBF
    'clf__class_weight': Categorical(['balanced']),
    'clf__shrinking': Categorical([True]),
},








         'LogisticRegression': {
   'clf__C': Real(1e-3, 1e3, prior='log-uniform', name='C'),  # Broader range
    'clf__penalty': Categorical(['l2'], name='penalty'),
    'clf__solver': Categorical(['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'], name='solver'),
    'clf__tol': Real(1e-6, 1e-2, prior='log-uniform', name='tol'),  # More precision
    'clf__fit_intercept': Categorical([True, False], name='fit_intercept'),
    'clf__class_weight': Categorical(['balanced', None], name='class_weight'),
    'clf__l1_ratio': Real(0.0, 1.0, name='l1_ratio'),  # Used only if penalty='elasticnet'
    'clf__max_iter': Categorical([100, 200, 300, 500, 1000], name='max_iter'),
    'clf__warm_start': Categorical([True, False], name='warm_start'),
    'clf__dual': Categorical([False], name='dual')  #
    },

    'RidgeClassifier': {
        'clf__alpha': Real(0.1, 100.0, prior='log-uniform'),
        'clf__solver': Categorical(['auto', 'sag', 'saga', 'lsqr'])
    },

    'CalibratedRidge': {
    'clf__estimator__alpha': Real(1e-4, 1e2, prior='log-uniform'),
    'clf__estimator__solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
    'clf__estimator__tol': Real(1e-5, 1e-2, prior='log-uniform'),
    'clf__estimator__fit_intercept': Categorical([True, False]),
    'clf__estimator__max_iter': Integer(1000, 10000),
},

'CalibratedLinearSVC': {
    'clf__estimator__C': Real(1e-3, 1e2, prior='log-uniform'),
    'clf__estimator__tol': Real(1e-5, 1e-1, prior='log-uniform'),
    'clf__estimator__loss': Categorical(['squared_hinge']),
    'clf__estimator__class_weight': Categorical([None, 'balanced']),
    'clf__estimator__max_iter': Integer(1000, 10000),
},

    'GaussianNB': {
        'clf__var_smoothing': Real(1e-12, 1e-6, prior='log-uniform')
    },
    'QDA': {
        'clf__reg_param': Real(0.0, 1.0, prior='uniform')
    },

      'AdaBoost': {
    'clf__n_estimators': Integer(50, 200, name='n_estimators'),  # Increased range, added name
    'clf__learning_rate': Real(0.01, 1.0, prior='log-uniform', name='learning_rate'),  # Added name
    'clf__algorithm': Categorical(['SAMME'], name='algorithm'), # Added algorithm choice, added name
    'clf__random_state': Integer(1, 10, name='random_state') #Added random_state, added name.
     },


   'SGDClassifier': {
        'clf__alpha': Real(1e-5, 1e-1, prior='log-uniform'),
        'clf__loss': Categorical(['hinge', 'log_loss', 'modified_huber']),
        'clf__penalty': Categorical(['l2', 'l1', 'elasticnet']),
        'clf__max_iter': Integer(500, 2000)
    }
}
best_estimators = {}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTuning {name}...")

    pipeline = Pipeline([
        ('clf', model)
    ])


    if name in params:
        search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=params[name],
            n_iter=50,
            scoring='balanced_accuracy',
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train_rfe, y_train)
        # search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_
        print(f"Best Parameters: {search.best_params_}")
    else:
        # pipeline.fit(X_train_rfe, y_train)
        pipeline.fit(X_train_rfe, y_train)
        best_estimators[name] = pipeline





def plot_learning_curve(model, X, y, cv=cv, scoring='balanced_accuracy', train_sizes=np.linspace(0.1, 1.0, 5),
                        model_name='Model'):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring,
                                                            train_sizes=train_sizes)

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Accuracy')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='test Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import numpy as np
# Evaluate each tuned model using cross-validation
results = []
names = []

for name, model in best_estimators.items():
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_rfe, y_train, cv=kfold, scoring='balanced_accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')
    plot_learning_curve(model, X_train_rfe, y_train, model_name=name)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from docx import Document
from docx.shared import Inches
from sklearn.model_selection import StratifiedKFold

# ----------------------------
# 1️⃣ Threshold Optimization (Leakage-Free)
# ----------------------------
def find_best_threshold(model, X_val, y_val):
    """Find threshold that maximizes balanced accuracy (no leakage)"""
    if not hasattr(model, "predict_proba"):
        return 0.5

    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh, best_bal_acc = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = t

    return best_thresh

def estimate_threshold_cv(model, X, y, cv_splits=5):
    """Estimate threshold via internal cross-validation (uses only training data)"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    thresholds = []
    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model.fit(X_train_cv, y_train_cv)
        t = find_best_threshold(model, X_val_cv, y_val_cv)
        thresholds.append(t)
    return np.mean(thresholds)

# ----------------------------
# 2️⃣ Evaluation Visualization
# ----------------------------
def plot_model_evaluation(model, X, y, model_name, feature_names=None, threshold=0.5):
    """Plot confusion matrix, ROC, PR, and feature importance"""
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else model.predict(X)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Model: {model_name} (Threshold={threshold:.2f})", fontsize=16)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, cmap='winter', annot=True, fmt='d',
                xticklabels=['Healthy', 'Patients'],
                yticklabels=['Healthy', 'Patients'], ax=axs[0])
    axs[0].set_title("Confusion Matrix")

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        axs[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axs[1].plot([0, 1], [0, 1], 'k--')
        axs[1].set_title('ROC Curve')
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].legend(loc='lower right')

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        avg_prec = average_precision_score(y, y_proba)
        axs[2].plot(recall, precision, label=f'AP = {avg_prec:.2f}')
        axs[2].set_title('Precision-Recall Curve')
        axs[2].set_xlabel('Recall')
        axs[2].set_ylabel('Precision')
        axs[2].legend(loc='lower left')
    else:
        axs[1].axis('off')
        axs[2].axis('off')

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = feature_names if feature_names is not None else [f'Feature {i}' for i in range(len(importances))]
        sorted_names = np.array(names)[indices]
        axs[3].barh(sorted_names[::-1], importances[indices][::-1])
        axs[3].set_title("Feature Importance")
        axs[3].set_xlabel("Importance")
    else:
        axs[3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{model_name}_evaluation.png", dpi=300)
    plt.show()

    return y_pred

# ----------------------------
# 3️⃣ Evaluation Loop (Leak-Free)
# ----------------------------
results = {}
doc = Document()
doc.add_heading('Classification Report Summary (Leak-Free Threshold)', level=1)

for name, model in best_estimators.items():
    print(f"\n🔍 Evaluating {name}...")

    # Fit model on full training data
    model.fit(X_train_rfe, y_train)

    # Find best threshold (only using training data or CV)
    best_thresh = estimate_threshold_cv(model, X_train_rfe, y_train.values.ravel(), cv_splits=5)

    # Apply to test set (no leakage)
    y_test_pred = plot_model_evaluation(
        model, X_test_rfe, y_test,
        model_name=name,
        feature_names=selected_features,
        threshold=best_thresh
    )

    # Compute metrics
    report = classification_report(y_test, y_test_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']

    results[name] = {
        'Accuracy': accuracy,
        'Balanced Accuracy': bal_acc,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1-Score (Macro)': f1_macro,
        'Best Threshold (train)': best_thresh
    }

    print(f"✅ {name} — Test Balanced Acc: {bal_acc:.3f}, Train-based Threshold: {best_thresh:.2f}")

    # Add to Word Report
    doc.add_heading(name, level=2)
    table = doc.add_table(rows=1, cols=len(results[name]) + 1)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Metric"
    for i, col_name in enumerate(results[name].keys(), start=1):
        hdr_cells[i].text = col_name
    row_cells = table.add_row().cells
    row_cells[0].text = name
    for i, value in enumerate(results[name].values(), start=1):
        row_cells[i].text = f"{value:.4f}"
    doc.add_picture(f"{name}_evaluation.png", width=Inches(6))

# Save Word document
doc.save(r"D:\model_results_structural_LeakFree.docx")
print("\n✅ Results saved to 'model_results_structural_LeakFree.docx'")

# Convert to DataFrame for comparison
df_results = pd.DataFrame(results).T
df_results = df_results.sort_values(by="Balanced Accuracy", ascending=False)
print("\nClassification Report Summary (Leak-Free):")
print(df_results)
