from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

import joblib
import numpy as np
import pandas as pd
import shap
from src.scrapper import scrape_twitter_user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load your trained model
model = joblib.load(str(BASE_DIR / "models" / "xgboost_model.pkl"))

# These must match EXACTLY the features you trained on

FEATURE_COLUMNS = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "verified",
    "default_profile_image",
    "account_age_days",
    "followers_friends_ratio",
    "statuses_per_day",
    "favourites_per_status",   # 👈 FIXED POSITION
    "log_followers",
    "log_friends",
    "log_statuses"
]


FEATURE_LABELS = {
    "followers_count":         "Follower count",
    "friends_count":           "Following count",
    "statuses_count":          "Tweet count",
    "favourites_count":        "Favorites count",
    "verified":                "Verified status",
    "default_profile_image":   "Default profile image",
    "account_age_days":        "Account age (days)",
    "followers_friends_ratio": "Follower/Following ratio",
    "statuses_per_day":        "Tweets per day",
    "favourites_per_status":   "Favorites per tweet",
    "log_followers":           "Follower count (log scale)",
    "log_friends":             "Following count (log scale)",
    "log_statuses":            "Tweet count (log scale)"
}

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/predict")
def predict(username: str):
    try:
        # 1. Scrape profile
        data = scrape_twitter_user(username)
        profile  = data["profile"]
        features = data["features"]

        # 2. Build feature vector in correct order
        feature_values = [features[col] for col in FEATURE_COLUMNS]
        print("FEATURE VALUES:", feature_values)
        X = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

        # 3. Predict
        prob      = model.predict_proba(X)[0]
        bot_prob  = float(prob[1])
        threshold = 0.5
        prediction = "Bot" if bot_prob >= threshold else "Human"

        # 4. Feature importance (using SHAP for per-user explanation)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For XGBoost binary classification, shap_values is usually (n_samples, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Get top 5 features by absolute impact magnitude
        sorted_idx = np.argsort(np.abs(sv))[::-1][:5]

        explanation = []
        for idx in sorted_idx:
            col   = FEATURE_COLUMNS[idx]
            val   = feature_values[idx]
            impact = sv[idx]
            label = FEATURE_LABELS.get(col, col)

            if impact > 0:
                direction   = "increases"
                explanation_text = f"{label} increases the likelihood of being a bot"
            else:
                direction   = "decreases"
                explanation_text = f"{label} reduces the likelihood of being a bot"

            explanation.append({
                "name":        label,
                "value":       round(val, 2) if isinstance(val, (float, np.floating)) else val,
                "direction":   direction,
                "explanation": explanation_text,
            })

        # 5. Final insight text
        top_feature = explanation[0]["name"] if explanation else "unknown feature"
        risk        = "High Risk" if bot_prob > 0.7 else "Medium Risk" if bot_prob > 0.4 else "Low Risk"

        insight = (
            f"This account appears to be a {'bot' if prediction=='Bot' else 'human'} account "
            f"based on its behavior. There is a {round(bot_prob*100, 1)}% probability that it "
            f"behaves like a bot. The strongest signal was '{top_feature}'. "
            f"Overall, this account falls under '{risk}', suggesting "
            f"{'high caution' if risk=='High Risk' else 'moderate caution' if risk=='Medium Risk' else 'low concern'}."
        )

        return {
            "prediction":    prediction,
            "bot_probability": bot_prob,
            "profile":       profile,
            "features":      explanation,
            "insight":       insight,
        }

    except Exception as e:
        return {"error": str(e)}
        