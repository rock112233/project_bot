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
    "default_profile_image",
    "account_age_days",
    "followers_friends_ratio",
    "statuses_per_day",
    "favourites_per_status",
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

        # Layman-language explanation map — returns (text, sentiment)
        # sentiment: "good" = human indicator, "bad" = bot indicator
        def _followers(v, d):
            n = int(v)
            if n > 1000: return f"Has {n:,} followers — a strong follower base signals genuine popularity", "good"
            elif n < 10: return f"Has only {n} followers — very few followers is a common bot trait", "bad"
            else: return f"Has {n} followers — a moderate count, within normal range", "neutral"

        def _friends(v, d):
            n = int(v)
            if n == 0: return f"Following 0 accounts — not following anyone is highly suspicious", "bad"
            elif n > 500: return f"Following {n:,} accounts — mass-following can be a bot tactic", "bad"
            else: return f"Following {n} accounts — a normal following count indicates genuine activity", "good"

        def _statuses(v, d):
            n = int(v)
            if n == 0: return f"Posted 0 tweets — zero activity is a classic bot placeholder sign", "bad"
            elif n > 50000: return f"Posted {n:,} tweets — extremely high tweet volume can indicate automation", "bad"
            else: return f"Posted {n:,} tweets — a reasonable tweet history suggests human behavior", "good"

        LAYMAN_EXPLAIN = {
            "followers_count": _followers,
            "friends_count": _friends,
            "statuses_count": _statuses,
            "favourites_count": lambda v, d: (
                (f"Liked 0 posts — no likes is uncommon for a real user", "bad") if int(v) == 0
                else (f"Liked {int(v):,} posts — liking content shows genuine interest", "good")
            ),
            "default_profile_image": lambda v, d: (
                ("Still using default avatar — bots often skip customization", "bad") if v == 1
                else ("Has a custom profile picture — a good human indicator", "good")
            ),
            "account_age_days": lambda v, d: (
                (f"Account is {int(v):,} days old — an older, established account is more trustworthy", "good") if int(v) > 1000
                else (f"Account is {int(v):,} days old — a newer account raises suspicion", "bad") if int(v) < 180
                else (f"Account is {int(v):,} days old — moderate age, within normal range", "neutral")
            ),
            "followers_friends_ratio": lambda v, d: (
                (f"Follower-to-following ratio is {round(v, 2)} — an extreme ratio raises concerns", "bad") if v > 10000 or v < 0.01
                else (f"Follower-to-following ratio is {round(v, 2)} — a healthy ratio suggests organic growth", "good")
            ),
            "statuses_per_day": lambda v, d: (
                (f"Tweets {round(v, 1)} times/day — zero daily activity is suspicious", "bad") if v == 0
                else (f"Tweets {round(v, 1)} times/day — abnormally high posting rate suggests automation", "bad") if v > 50
                else (f"Tweets {round(v, 1)} times/day — a normal posting frequency", "good")
            ),
            "favourites_per_status": lambda v, d: (
                (f"Averages 0 likes per tweet — zero engagement is a red flag", "bad") if v == 0
                else (f"Averages {round(v, 2)} likes per tweet — very low engagement is concerning", "bad") if v < 0.05
                else (f"Averages {round(v, 2)} likes per tweet — healthy engagement ratio", "good")
            ),
            "log_followers": lambda v, d: (
                (f"Follower magnitude: {round(v, 2)} — strong follower presence boosts human confidence", "good") if v > 8
                else (f"Follower magnitude: {round(v, 2)} — low follower magnitude is a bot indicator", "bad") if v < 4
                else (f"Follower magnitude: {round(v, 2)} — follower scale is within normal range", "neutral")
            ),
            "log_friends": lambda v, d: (
                (f"Following magnitude: {round(v, 2)} — zero following is an anomaly", "bad") if v == 0
                else (f"Following magnitude: {round(v, 2)} — abnormal following pattern detected", "bad") if d == 'increases'
                else (f"Following magnitude: {round(v, 2)} — normal following behavior", "good")
            ),
            "log_statuses": lambda v, d: (
                (f"Tweet volume score: {round(v, 2)} — unusual tweeting volume detected", "bad") if d == 'increases'
                else (f"Tweet volume score: {round(v, 2)} — healthy tweet volume", "good")
            ),
        }

        # Human-readable labels for raw values
        VALUE_LABELS = {
            "followers_count": lambda v: f"{int(v):,} followers",
            "friends_count": lambda v: f"{int(v):,} following",
            "statuses_count": lambda v: f"{int(v):,} tweets",
            "favourites_count": lambda v: f"{int(v):,} likes",
            "default_profile_image": lambda v: "Yes" if v == 1 else "No",
            "account_age_days": lambda v: f"{int(v):,} days",
            "followers_friends_ratio": lambda v: f"{round(v, 2)} ratio",
            "statuses_per_day": lambda v: f"{round(v, 1)} tweets/day",
            "favourites_per_status": lambda v: f"{round(v, 2)} likes/tweet",
            "log_followers": lambda v: f"log({int(round(np.exp(v))):,})",
            "log_friends": lambda v: f"log({int(round(np.exp(v))):,})",
            "log_statuses": lambda v: f"log({int(round(np.exp(v))):,})",
        }

        explanation = []
        for idx in sorted_idx:
            col   = FEATURE_COLUMNS[idx]
            val   = feature_values[idx]
            impact = sv[idx]
            label = FEATURE_LABELS.get(col, col)
            direction = "increases" if impact > 0 else "decreases"

            # Generate layman explanation + sentiment
            explain_fn = LAYMAN_EXPLAIN.get(col)
            if explain_fn:
                result = explain_fn(val, direction)
                if isinstance(result, tuple):
                    explanation_text, sentiment = result
                else:
                    explanation_text, sentiment = result, ("bad" if direction == "increases" else "good")
            else:
                explanation_text = f"{label} {'raises' if direction == 'increases' else 'lowers'} bot suspicion"
                sentiment = "bad" if direction == "increases" else "good"

            # Human readable value label
            vl_fn = VALUE_LABELS.get(col)
            value_label = vl_fn(val) if vl_fn else str(round(val, 2))

            explanation.append({
                "name":        label,
                "value":       round(val, 2) if isinstance(val, (float, np.floating)) else val,
                "value_label": value_label,
                "direction":   direction,
                "sentiment":   sentiment,
                "impact":      round(float(impact), 4),
                "explanation": explanation_text,
            })

        # Build SHAP waterfall data (all 12 features, sorted by impact)
        all_shap = []
        for i in range(len(FEATURE_COLUMNS)):
            all_shap.append({
                "name": FEATURE_LABELS.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]),
                "impact": round(float(sv[i]), 4),
            })
        all_shap.sort(key=lambda x: x["impact"])  # negative (human) first, positive (bot) last

        # 5. Rich final insight
        top_feature = explanation[0]["name"] if explanation else "unknown feature"
        risk = "High Risk" if bot_prob > 0.7 else "Medium Risk" if bot_prob > 0.4 else "Low Risk"

        p = profile
        followers = p.get("followers", 0)
        following = p.get("following", 0)
        tweets = p.get("tweets", 0)
        has_bio = p.get("has_bio", False)
        joined = p.get("joined", "unknown")

        insight_parts = []
        insight_parts.append(
            f"After analyzing @{p.get('username', username)}'s behavioral patterns, "
            f"MASKOFF assigns a {round(bot_prob*100, 1)}% bot probability — classified as '{risk}'."
        )

        if prediction == "Bot":
            reasons = []
            if followers < 10:
                reasons.append(f"only {followers} followers")
            if tweets == 0:
                reasons.append("zero tweets posted")
            if not has_bio:
                reasons.append("no bio description")
            if following == 0:
                reasons.append("not following anyone")
            if reasons:
                insight_parts.append(f"Key concerns include: {', '.join(reasons)}.")
            insight_parts.append(
                f"The strongest signal was '{top_feature}'. "
                f"These patterns closely match bot behavior observed in our training data of 300,000 accounts."
            )
        else:
            strengths = []
            if followers > 100:
                strengths.append(f"a solid follower base of {followers:,}")
            if tweets > 50:
                strengths.append(f"an active tweet history ({tweets:,} tweets)")
            if has_bio:
                strengths.append("a completed bio")
            if strengths:
                insight_parts.append(f"Positive signals include: {', '.join(strengths)}.")
            insight_parts.append(
                f"The account joined on {joined} and shows organic engagement patterns "
                f"consistent with genuine human activity."
            )

        insight = " ".join(insight_parts)

        # 6. Derived stats for profile card
        age_days = features.get("account_age_days", 0)
        years = int(age_days // 365)
        months = int((age_days % 365) // 30)
        age_str = f"{years}y {months}m" if years > 0 else f"{months} months"

        derived_stats = {
            "account_age_human": age_str,
            "tweets_per_day": round(features.get("statuses_per_day", 0), 2),
            "engagement_rate": round(features.get("favourites_per_status", 0), 2),
            "follower_ratio": round(features.get("followers_friends_ratio", 0), 1),
            "bio": profile.get("bio", ""),
        }

        return {
            "prediction":    prediction,
            "bot_probability": bot_prob,
            "profile":       profile,
            "features":      explanation,
            "insight":       insight,
            "shap_waterfall": all_shap,
            "derived_stats": derived_stats,
        }

    except Exception as e:
        return {"error": str(e)}
        