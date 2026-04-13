import ijson
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report

print("Loading labels...")
labels_df = pd.read_csv('dataset/label.csv')
# Create a dictionary for fast lookup O(1)
label_map = dict(zip(labels_df['id'], labels_df['label']))

# Features to extract
data = []
count = 0

print("Streaming user.json...")
with open('dataset/user.json', 'rb') as f:
    # `ijson.items` yields JSON array elements one by one
    for user in ijson.items(f, 'item'):
        uid = user.get('id')
        if uid not in label_map:
            continue
            
        label = 1 if label_map[uid] == 'bot' else 0
        
        metrics = user.get('public_metrics', {})
        f_count = metrics.get('followers_count', 0)
        fg_count = metrics.get('following_count', 0)
        t_count = metrics.get('tweet_count', 0)
        fav_count = metrics.get('listed_count', 0) # Fallback if likes NA
        
        dpi = 1 if user.get('profile_image_url') and 'default_profile' in user.get('profile_image_url', '') else 0
        
        # Calculate age relative to 2022 (when dataset was published) to preserve ratio meanings
        created_at_str = user.get('created_at')
        try:
            # "2013-05-02 21:28:39+00:00"
            created_dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S%z")
            # Dataset cutoff is around 2022
            cutoff_dt = datetime(2022, 1, 1, tzinfo=timezone.utc)
            age_days = (cutoff_dt - created_dt).days
            age_days = max(age_days, 1)
        except:
            age_days = 365
            
        # We explicitly skip 'verified' to prevent target leakage!
            
        row = {
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
        }
        data.append(row)
        
        count += 1
        if count % 20000 == 0:
            print(f"Processed {count} users...")
        if count >= 300000: # Limit to 300k to save time and memory, but enough for highly robust training
            break

print("Creating dataframe...")
df = pd.DataFrame(data)

X = df.drop('label', axis=1)
y = df['label']

print(f"Dataset shape: {X.shape}")
print(f"Class distribution before SMOTE: \\n{y.value_counts()}")

print("Balancing classes using SMOTE + Tomek Links...")
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

print("Training XGBoost Classifier without 'verified' column...")
# Adjusting max_depth to force it to learn behavioral parameters
model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Evaluation metrics:")
print(classification_report(y_test, preds))
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

print("Feature Importances:")
importances = list(zip(X.columns, model.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importances:
    print(f"{feat}: {imp:.4f}")

joblib.dump(model, 'MASKOFF_Project/models/xgboost_model.pkl')
print("Saved new unbiased model!")
