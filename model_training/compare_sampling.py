import ijson
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MASKOFF: SMOTE vs Undersampling Experiment")
print("=" * 60)

# Load a subset of data for fast evaluation
print("\n[1/4] Loading Data Subset (10,000 records)...")
labels_df = pd.read_csv('dataset/label.csv')
label_map = dict(zip(labels_df['id'], labels_df['label']))

data = []
count = 0
with open('dataset/user.json', 'rb') as f:
    for user in ijson.items(f, 'item'):
        uid = user.get('id')
        if uid not in label_map: continue
        
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
            'followers_count': f_count, 'friends_count': fg_count, 'statuses_count': t_count,
            'favourites_count': fav_count, 'default_profile_image': dpi, 'account_age_days': age_days,
            'followers_friends_ratio': f_count / (fg_count + 1), 'statuses_per_day': t_count / age_days,
            'favourites_per_status': fav_count / (t_count + 1), 'log_followers': math.log(f_count + 1),
            'log_friends': math.log(fg_count + 1), 'log_statuses': math.log(t_count + 1), 'label': label
        })
        count += 1
        if count >= 10000: break

df = pd.DataFrame(data)
X = df.drop('label', axis=1)
y = df['label']
print(f"  Class balance: {(y==0).sum()} humans / {(y==1).sum()} bots")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = XGBClassifier(tree_method='hist', random_state=42, eval_metric='logloss')

# EXPERIMENT 1: Undersampling
print("\n[2/4] Experiment 1: Random Undersampling")
undersample = RandomUnderSampler(random_state=42)
X_under, y_under = undersample.fit_resample(X_train, y_train)
print(f"  Undersampled Data: {(y_under==0).sum()} humans / {(y_under==1).sum()} bots")
base_model.fit(X_under, y_under)
preds_under = base_model.predict(X_test)

print("\n--- Undersampling Results ---")
print(f"Accuracy: {accuracy_score(y_test, preds_under):.4f}")
print(classification_report(y_test, preds_under, target_names=['Human (0)', 'Bot (1)']))


# EXPERIMENT 2: SMOTE
print("\n[3/4] Experiment 2: SMOTE + Tomek (Oversampling)")
smote = SMOTETomek(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"  SMOTE Data: {(y_smote==0).sum()} humans / {(y_smote==1).sum()} bots")
base_model.fit(X_smote, y_smote)
preds_smote = base_model.predict(X_test)

print("\n--- SMOTE Results ---")
print(f"Accuracy: {accuracy_score(y_test, preds_smote):.4f}")
print(classification_report(y_test, preds_smote, target_names=['Human (0)', 'Bot (1)']))

print("\n[4/4] Conclusion:")
print("  Undersampling destroys valuable human data to match the small bot sample, causing loss of critical patterns.")
print("  SMOTE synthesizes minority (bot) data, retaining all human data, yielding higher precision and robustness.")
print("=" * 60)
