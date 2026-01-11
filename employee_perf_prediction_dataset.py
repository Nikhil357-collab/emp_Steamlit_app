import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("C:/Users/NIKHIL/Downloads/vscode.python/data/employee_perf_prediction_dataset.csv")

# Unique employee check (prevents leakage)
assert df['employee_id'].is_unique, "Duplicate employee_id detected"

TARGET = "perf_band_next"


# 2. Sanity checks
 
rules = {
    'experience_years': (0, 45),
    'on_time_delivery_rate': (0, 1),
    'manager_score_mid_year': (0, 5),
    'peer_feedback_score': (0, 5),
    'avg_task_delay_days': (-10, 60)}

for col, (lo, hi) in rules.items():
    outliers = df[(df[col] < lo) | (df[col] > hi)]
    print(f"{col} out_of_range: {len(outliers)}")


# 3. Handle rare classes (required for stratification)

class_counts = df['perf_band_next'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['perf_band_next'].isin(valid_classes)]

# 4. Split features and target
X = df.drop(columns=['perf_band_next'])
y = df['perf_band_next']

# 5. Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=13,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Train class distribution:\n", y_train.value_counts(normalize=True))


from sklearn.feature_selection import mutual_info_classif
# Create a "train" dataframe in the same form your code expects
train = X_train.copy()
train["perf_band_next"] = y_train  # add target back as a column
X =train.drop(columns=['perf_band_next'])
y = train['perf_band_next']
num_cols = X.select_dtypes(include='number').columns
mi = mutual_info_classif(X[num_cols].fillna(0), y)
pd.Series(mi, index=num_cols).sort_values(ascending=False).head(10)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 5) Column groups
cat_cols = X_train.select_dtypes(include= "category").columns
num_cols = X_train.select_dtypes(include="number").columns

# 6) Preprocessing pipelines
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop"
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Train class distribution:\n", y_train.value_counts(normalize=True))



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

pipe_log = Pipeline([('pre', preprocess),
                     ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))])

pipe_rf  = Pipeline([('pre', preprocess),
                     ('clf', RandomForestClassifier(class_weight='balanced', random_state=13))])

param_rf = {
  'clf__n_estimators':[200,400],
  'clf__max_depth':[8,12,None],
  'clf__min_samples_leaf':[1,3,5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
gs = GridSearchCV(pipe_rf, param_grid=param_rf, scoring='f1_macro', cv=cv, n_jobs=-1)#trains many versions of the RandomForest pipeline using all combinations from param_rf.
gs.fit(X, y)
best_model = gs.best_estimator_ # the best full pipeline (preprocessing + best RF settings).
print("CV F1_macro:", gs.best_score_)
print("df:", df.shape)


from sklearn.metrics import classification_report, confusion_matrix
# Evaluate on test set classification_report prints precision, recall, F1-score for each class (H, M, etc.).
#confusion_matrix prints a table of correct vs wrong predictions.


from sklearn.metrics import classification_report, confusion_matrix

test = X_test.copy()
test['perf_band_next'] = y_test
X_test, y_test = test.drop(columns=['perf_band_next']), test['perf_band_next']

y_pred = best_model.predict(X_test)
y_proba = getattr(best_model.named_steps['clf'], "predict_proba", lambda z: None)(best_model.named_steps['pre'].transform(X_test))

print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))
#Accuracy = 0.900: overall, 90% of test predictions were correct.​
#Macro avg F1 = 0.899: average F1 across classes giving equal weight to H and M (good when classes are not perfectly 



import numpy as np
from sklearn.inspection import permutation_importance
r = permutation_importance(best_model, X_test, y_test, n_repeats=5, scoring='f1_macro', random_state=13)
imp = pd.Series(r.importances_mean)
print(imp.sort_values(ascending=False).head(15))
print("Importances length:", len(r.importances_mean))#r.importances_mean: mean importance for each feature.
print("Feature names length:", len(best_model.named_steps['pre'].get_feature_names_out()))

#####...........................
import pandas as pd
import numpy as np

def group_report(df_true, y_pred, group_col, positive_class='H'):
    out = []
    for g in df_true[group_col].dropna().unique():
        idx = (df_true[group_col] == g).values
        sel = (pd.Series(y_pred).values[idx] == positive_class).mean()
        out.append((g, round(sel, 3)))
    return pd.DataFrame(out, columns=[group_col, 'selection_rate'])

# Create test dataframe with gender column
test = X_test.copy()
test['_y'] = y_test.values

rep = group_report(test, y_pred, 'gender', positive_class='H')
print(rep)

#
def group_report(df_true, y_pred, group_col, positive_class='H'):
    out = []
    for g in df_true[group_col].dropna().unique():
        idx = (df_true[group_col] == g).values
        sel = (pd.Series(y_pred).values[idx] == positive_class).mean()
        count = idx.sum()
        out.append((g, round(sel, 3), count))
    
    result = pd.DataFrame(out, columns=[group_col, 'selection_rate', 'count'])
    
    # Calculate and print min/max ratio
    if len(result) > 1:
        min_rate = result['selection_rate'].min()
        max_rate = result['selection_rate'].max()
        ratio = min_rate / max_rate if max_rate > 0 else 0
        print(f"\nMin/Max ratio: {ratio:.3f}")
        if ratio < 0.80:
            print("⚠️ WARNING: Fails 80% rule (potential disparate impact)")
    
    return result

# Create test dataframe with gender column
test = X_test.copy()
test['_y'] = y_test.values

rep = group_report(test, y_pred, 'gender', positive_class='H')
print(rep)


def recommend_actions(features):
    actions = []

    if features['on_time_delivery_rate'] < 0.75:
        actions.append("Enroll in sprint planning/time-management workshop")

    if features['bug_count'] > 15 and features['code_review_score'] < 3.5:
        actions.append("Pair programming + code quality training")

    if features['training_hours'] < 50:
        actions.append("Assign targeted certification (e.g., AWS CCP / Azure DP-900)")

    return actions[:2]   # top 2 actions
