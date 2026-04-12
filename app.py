from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import joblib, json, numpy as np
from scrapper import scrape_twitter_user

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and features once at startup
model = joblib.load("xgboost_model.pkl")
with open("feature_cols.json") as f:
    feature_cols = json.load(f)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request):
    body = await request.json()
    username = body.get("username", "").strip().lstrip("@")

    try:
        # Step 1: Scrape user data
        user_data = scrape_twitter_user(username)

        # Step 2: Build feature vector (must match training order)
        row = [user_data.get(col, 0) for col in feature_cols]
        X = np.array(row).reshape(1, -1)

        # Step 3: Predict
        prob = model.predict_proba(X)[0]
        bot_prob  = round(float(prob[1]) * 100, 1)
        human_prob = round(float(prob[0]) * 100, 1)
        label = "Bot" if bot_prob > 50 else "Human"
        risk  = "High Risk" if bot_prob > 80 else ("Medium Risk" if bot_prob > 50 else "Low Risk")

        # Step 4: Feature importance (explainability)
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:5]

        explanations = []
        for feat, score in feat_imp:
            val = user_data.get(feat, 0)
            direction = "increases" if bot_prob > 50 else "reduces"
            explanations.append({
                "feature": feat.replace("_", " "),
                "value": round(val, 2),
                "impact": direction,
                "score": round(float(score), 4)
            })

        insight = (
            f"This account appears to be a {'bot' if label=='Bot' else 'human'} account "
            f"based on its behavior. There is a {bot_prob}% probability that it behaves "
            f"like a bot. The strongest signal was '{feat_imp[0][0].replace('_',' ')}'. "
            f"Overall, this account falls under '{risk}'."
        )

        return JSONResponse({
            "label": label,
            "risk": risk,
            "bot_prob": bot_prob,
            "human_prob": human_prob,
            "profile": {
                "username": "@" + username,
                "display_name": user_data.get("display_name", ""),
                "followers": user_data.get("followers", 0),
                "following": user_data.get("following", 0),
                "tweets": user_data.get("tweets", 0),
                "bio": user_data.get("bio", ""),
                "location": user_data.get("location", ""),
            },
            "explanations": explanations,
            "insight": insight
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)