"""Run the analysis from the repository root."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

SEED = 42
ROOT = Path.cwd()
for directory in ['results', 'figures']:
    (ROOT / directory).mkdir(exist_ok=True)
data_path = ROOT / 'data' / 'heart.csv'
if not data_path.exists():
    raise FileNotFoundError('Place heart.csv in data/ or run python download_data.py first.')
data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
df = pd.read_csv(data_path)
print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns')
print('SHA-256:', data_hash)
assert set(df['HeartDisease'].unique()) == {0, 1}
assert len(df) == 918, 'Expected Kaggle dataset version 1 (918 rows).'
print('Exact duplicate rows:', int(df.duplicated().sum()))


numeric = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
X = df[numeric + categorical].copy()
y = df['HeartDisease'].copy()
for column in ['Cholesterol', 'RestingBP']:
    X[column] = X[column].replace(0, np.nan)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
assert set(X_train.index).isdisjoint(X_test.index)
print('Training rows:', len(X_train), '| Test rows:', len(X_test))
print('Training positive fraction:', round(float(y_train.mean()), 4))

summary = X_train[numeric].describe().T
summary.to_csv(ROOT / 'results' / 'training_summary.csv')
print(summary.round(2).to_string())


preprocessor = ColumnTransformer([
    ('numeric', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())]), numeric),
    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
])
pipeline = Pipeline([('preprocessor', preprocessor),
                     ('model', DummyClassifier(strategy='most_frequent'))])
grid = [
    {'model': [DummyClassifier(strategy='most_frequent')]},
    {'model': [DecisionTreeClassifier(random_state=SEED, min_samples_leaf=10)],
     'model__max_depth': [3, 4]},
    {'model': [RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1)],
     'model__min_samples_leaf': [3, 5]},
    {'model': [XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='logloss', random_state=SEED, n_jobs=1)]},
    {'model': [LogisticRegression(solver='liblinear', max_iter=5000, random_state=SEED)],
     'model__C': [0.1, 1.0, 10.0]},
    {'model': [LogisticRegression(solver='liblinear', l1_ratio=1.0,
                                  max_iter=5000, random_state=SEED)],
     'model__C': [0.1, 1.0, 10.0]}
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = GridSearchCV(pipeline, grid, cv=cv, scoring={
    'roc_auc':'roc_auc', 'accuracy':'accuracy', 'f1':'f1', 'recall':'recall'},
    refit='roc_auc', n_jobs=1, error_score='raise')
search.fit(X_train, y_train)

def model_label(model):
    if isinstance(model, LogisticRegression):
        return 'Logistic Regression (L1)' if model.l1_ratio == 1.0 else 'Logistic Regression (L2)'
    return type(model).__name__

rows = []
for i, params in enumerate(search.cv_results_['params']):
    rows.append({
        'model': model_label(params['model']),
        'parameters': json.dumps({k: v for k,v in params.items() if k != 'model'}),
        **{name: float(search.cv_results_[name][i]) for name in [
            'mean_test_roc_auc', 'std_test_roc_auc', 'mean_test_accuracy',
            'mean_test_f1', 'mean_test_recall']}})
cv_results = pd.DataFrame(rows).sort_values('mean_test_roc_auc', ascending=False, kind='stable')
cv_results.to_csv(ROOT / 'results' / 'cv_results.csv', index=False)
print(cv_results.round(4).to_string(index=False))
best_model = search.best_estimator_
winner = model_label(best_model.named_steps['model'])
print('Selected using CV:', winner)


def metrics(model):
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    return dict(accuracy=float(accuracy_score(y_test,pred)),
                precision=float(precision_score(y_test,pred,zero_division=0)),
                recall=float(recall_score(y_test,pred,zero_division=0)),
                f1=float(f1_score(y_test,pred,zero_division=0)),
                roc_auc=float(roc_auc_score(y_test,score)))

baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
holdout = pd.DataFrame([{'model':'Majority baseline', **metrics(baseline)},
                        {'model':winner, **metrics(best_model)}])
holdout.to_csv(ROOT / 'results' / 'holdout_metrics.csv', index=False)
print(holdout.round(4).to_string(index=False))
fig, axes = plt.subplots(1,2,figsize=(10,4))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0],
    display_labels=['No disease', 'Disease'], cmap='Blues', colorbar=False)
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1],name=winner)
axes[1].plot([0,1],[0,1],'--',color='gray')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'holdout_evaluation.png', dpi=160)
plt.show()

# Training-set permutation importance is descriptive, not an independent validation.
importance = permutation_importance(best_model, X_train, y_train,
    scoring='roc_auc', n_repeats=10, random_state=SEED, n_jobs=1)
feature_importance = pd.DataFrame({'feature':X_train.columns,
    'mean_auc_decrease':importance.importances_mean,
    'std_auc_decrease':importance.importances_std}).sort_values('mean_auc_decrease')
feature_importance.to_csv(ROOT / 'results' / 'training_permutation_importance.csv',index=False)
fig,ax = plt.subplots(figsize=(7,4.5))
ax.barh(feature_importance['feature'],feature_importance['mean_auc_decrease'],
        xerr=feature_importance['std_auc_decrease'],color='#356c91')
ax.set_xlabel('Training ROC-AUC decrease after permutation')
ax.set_title('Descriptive training-set feature importance')
fig.tight_layout()
fig.savefig(ROOT / 'figures' / 'feature_importance.png',dpi=160)
plt.show()

metadata = {
    'seed':SEED,'dataset':'fedesoriano/heart-failure-prediction','dataset_version':1,
    'data_sha256':data_hash,'rows':len(df),'train_rows':len(X_train),'test_rows':len(X_test),
    'selected_model':winner,'best_cv_roc_auc':float(search.best_score_),
    'selected_parameters':{k:v for k,v in search.best_params_.items() if k!='model'},
    'test_partition_previously_inspected':True,
    'python':platform.python_version(),
    'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','xgboost']}}
(ROOT/'results'/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
