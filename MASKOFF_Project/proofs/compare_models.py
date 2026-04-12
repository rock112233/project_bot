import ijson
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

print("Loading data...")
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
        
        try:
            from datetime import datetime, timezone
            dt = datetime.strptime(user.get('created_at'), "%Y-%m-%d %H:%M:%S%z")
            age_days = max((datetime(2022, 1, 1, tzinfo=timezone.utc) - dt).days, 1)
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
        if count >= 300000: break

df = pd.DataFrame(data)
X = df.drop('label', axis=1)
y = df['label']

print(f"Original Class Distribution: \\n{y.value_counts()}")

def evaluate_model(X_train, y_train, X_test, y_test, name):
    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    return {
        "Technique": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision (Bot)": precision_score(y_test, preds, pos_label=1),
        "Recall (Bot)": recall_score(y_test, preds, pos_label=1),
        "Precision (Human)": precision_score(y_test, preds, pos_label=0),
        "Recall (Human)": recall_score(y_test, preds, pos_label=0)
    }

# Ensure exactly the same test set is used for fairness
X_train_base, X_test, y_train_base, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training SMOTE...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_base, y_train_base)
res_smote = evaluate_model(X_smote, y_smote, X_test, y_test, "Our SMOTE")

print("Training Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_base, y_train_base)
res_rus = evaluate_model(X_rus, y_rus, X_test, y_test, "Current Undersampling")

results = pd.DataFrame([res_smote, res_rus])
print("\\n--- RESULTS TABLE ---")
print(results.to_string(index=False))

# Plotting the Graph
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision (Bot)', 'Recall (Bot)', 'Precision (Human)', 'Recall (Human)']
plot_data = pd.melt(results, id_vars='Technique', value_vars=metrics, var_name='Metric', value_name='Score')

sns.barplot(x='Metric', y='Score', hue='Technique', data=plot_data, palette=['#1DA1F2', '#FF5A5F'])
plt.title('MASKOFF: Modern SMOTE vs Modern Undersampling (Twibot-22 Dataset)', fontweight='bold')
plt.ylim(0, 1.05)
plt.ylabel('Score (0.0 to 1.0)')
plt.xticks(rotation=15)
plt.legend(title='Technique')

plt.tight_layout()
plt.savefig('MASKOFF_Project/docs/undersampling_vs_smote.png', dpi=300)
print("Graph saved to MASKOFF_Project/docs/undersampling_vs_smote.png")
