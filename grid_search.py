"""
MASKOFF GridSearch — GPU-Accelerated (RTX 3050)
Finds the mathematically optimal XGBoost hyperparameters for bot detection.
Runs SMOTE+Tomek balancing, then exhaustive GridSearchCV on GPU.
"""
import ijson
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math
import joblib
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("  MASKOFF GridSearch — GPU Accelerated (RTX 3050)")
print("=" * 60)

# ── Step 1: Load Data ──
print("\n[1/5] Loading labels...")
labels_df = pd.read_csv('dataset/label.csv')
label_map = dict(zip(labels_df['id'], labels_df['label']))

data = []
count = 0

print("[2/5] Streaming user.json...")
with open('dataset/user.json', 'rb') as f:
    for user in ijson.items(f, 'item'):
        uid = user.get('id')
        if uid not in label_map:
            continue
        label = 1 if label_map[uid] == 'bot' else 0

        metrics = user.get('public_metrics', {})
        f_count = metrics.get('followers_count', 0)
        fg_count = metrics.get('following_count', 0)
        t_count = metrics.get('tweet_count', 0)
        fav_count = metrics.get('listed_count', 0)

        dpi = 1 if user.get('profile_image_url') and 'default_profile' in user.get('profile_image_url', '') else 0

        created_at_str = user.get('created_at')
        try:
            created_dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S%z")
            cutoff_dt = datetime(2022, 1, 1, tzinfo=timezone.utc)
            age_days = max((cutoff_dt - created_dt).days, 1)
        except:
            age_days = 365

        data.append({
            'followers_count': f_count,
            'friends_count': fg_count,
            'statuses_count': t_count,
            'favourites_count': fav_count,
            'default_profile_image': dpi,
            'account_age_days': age_days,
            'followers_friends_ratio': f_count / (fg_count + 1),
            'statuses_per_day': t_count / age_days,
            'favourites_per_status': fav_count / (t_count + 1),
            'log_followers': math.log(f_count + 1),
            'log_friends': math.log(fg_count + 1),
            'log_statuses': math.log(t_count + 1),
            'label': label
        })

        count += 1
        if count % 50000 == 0:
            print(f"  Processed {count} users...")
        if count >= 300000:
            break

df = pd.DataFrame(data)
X = df.drop('label', axis=1)
y = df['label']

print(f"  Dataset: {X.shape[0]} users, {X.shape[1]} features")
print(f"  Class balance: {(y==0).sum()} humans / {(y==1).sum()} bots")

# ── Step 2: SMOTE + Tomek ──
print("\n[3/5] Balancing with SMOTE + Tomek Links...")
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)
print(f"  Balanced dataset: {X_res.shape[0]} samples")

# ── Step 3: GPU GridSearch ──
print("\n[4/5] Starting GPU-Accelerated GridSearch...")
print("  Device: NVIDIA RTX 3050 (CUDA)")
print("  Strategy: 3-Fold Stratified CV")

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3],
}

total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)

print(f"  Total combinations: {total_combos}")
print(f"  Total model trainings (x3 folds): {total_combos * 3}")
print(f"  Estimated time: ~1-2 hours on GPU\n")

base_model = XGBClassifier(
    device='cuda',
    tree_method='hist',
    random_state=42,
    eval_metric='logloss',
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

start_time = time.time()

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=1,  # GPU handles parallelism internally
    verbose=2,
    refit=True,
)

grid_search.fit(X_res, y_res)

elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)

# ── Step 4: Results ──
print("\n" + "=" * 60)
print("  GRIDSEARCH COMPLETE")
print("=" * 60)
print(f"  Time elapsed: {hours}h {minutes}m")
print(f"\n  Best Parameters:")
for k, v in grid_search.best_params_.items():
    print(f"    {k}: {v}")
print(f"\n  Best CV Accuracy: {grid_search.best_score_:.4f}")

# ── Step 5: Evaluate on holdout & save ──
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

print(f"\n  Holdout Test Accuracy: {accuracy_score(y_test, preds):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, preds))

print("Feature Importances:")
importances = list(zip(X.columns, best_model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importances:
    print(f"  {feat}: {imp:.4f}")

# Save the champion model
joblib.dump(best_model, 'MASKOFF_Project/models/xgboost_model.pkl')
print("\n  Champion model saved to MASKOFF_Project/models/xgboost_model.pkl")
print("=" * 60)
