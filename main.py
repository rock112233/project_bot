from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
import joblib
import numpy as np
import pandas as pd
from scrapper import scrape_twitter_user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load your trained model
model = joblib.load("xgboost_model.pkl")

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
    "followers_count":   "Follower count",
    "following_count":   "Following count",
    "tweet_count":       "Tweet count",
    "account_age_days":  "Account age (days)",
    "tweets_per_day":    "Tweets per day",
    "following_per_day": "Following per day",
    "followers_log":     "Follower count (log scale)",
    "following_log":     "Following count (log scale)",
    "tweets_log":        "Tweet count (log scale)",
    "ff_ratio":          "Follower/Following ratio",
    "bio_word_count":    "Bio word count",
    "verified":          "Verified account",
    "has_location":      "Has location",
    "has_bio":           "Has bio",
    "pinned_tweet":      "Has pinned tweet",
}

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

        # 4. Feature importance (from XGBoost)
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1][:5]  # top 5

        explanation = []
        for idx in sorted_idx:
            col   = FEATURE_COLUMNS[idx]
            val   = feature_values[idx]
            imp   = importances[idx]
            label = FEATURE_LABELS.get(col, col)

            if imp > 0 and bot_prob >= threshold:
                direction   = "increases"
                explanation_text = f"{label} strongly increases the likelihood of being a bot"
            else:
                direction   = "decreases"
                explanation_text = f"{label} strongly reduces the likelihood of being a bot"

            explanation.append({
                "name":        label,
                "value":       round(val, 2),
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
        