# MASKOFF: Explainable Bot Detection Platform — Final Project Report

## 1. Executive Summary
MASKOFF is an advanced, high-performance web platform utilizing Machine Learning (ML) and Explainable AI (XAI) to accurately detect automated bot profiles on Twitter (X). Designed with speed, precision, and transparency in mind, MASKOFF transitions the opaque "black box" nature of traditional machine learning into a user-friendly, completely transparent visual dashboard.

This document serves as the comprehensive architectural and analytical knowledge dump detailing the project's evolution, current methodology, and potential future scope.

---

## 2. Model Evolution: Why We Upgraded

### 2.1 The Original Model (Constraints & Limitations)
The initial model architecture we began with suffered from several critical bottlenecks common in early-stage ML projects:
* **Class Imbalance & Bias:** The original datasets heavily skewed toward "Human" accounts, causing the model to over-predict humanity and artificially inflate its accuracy metrics while missing sophisticated bots. (Addressed initially via basic resampling).
* **Feature Bloat:** The initial data ingestion pulled massive amounts of un-normalized text data causing sluggish prediction times.
* **The "Black Box" Problem:** It provided a binary *Bot* or *Human* verdict with zero justification. If an account was flagged incorrectly, there was no mathematical way to explain *why* the model made that decision to a non-technical user.

### 2.2 The Current Model (XGBoost + SHAP)
The current production model represents a massive leap in statistical capability and architectural resilience:
* **Algorithm Choice (XGBoost):** We pivoted to an Extreme Gradient Boosting (XGBoost) classifier. It natively handles non-linear relationships and sparse metadata significantly better than standard Deep Learning or basic Random Forests. 
* **Rigorous Hyperparameter Tuning:** The final model wasn't guessed; it was derived from an exhaustive Grid Search mapping across 864 unique hyperparameter combinations (learning rate, depth, estimators, subsampling).
* **Final Accuracy:** The validated model achieves peak accuracy of **92.58%**.
* **Engineered Behavioral Features:** Instead of analyzing heavy NLP text (which takes massive computational power), the model was restricted to exactly 12 normalized *behavioral metadata points* (e.g., Follower/Following Ratio, Logarithmic scale of follower counts, Account Age divided by total tweets). This restriction ensures the entire inference pipeline executes in **under 2 seconds**.
* **Explainable AI (XAI):** We integrated the **SHAP (SHapley Additive exPlanations)** framework. SHAP calculates the exact margin of impact every specific feature had on the final score. This mathematical matrix is then parsed by our custom backend "Value-Based Sentiment Engine" to generate human-readable explanations (e.g., "Account posts 0.0 tweets a day, raising bot suspicion").

---

## 3. Technology Stack & Operational Use Case

### 3.1 The Use Case
MASKOFF is designed for security researchers, social media managers, and everyday users who need to instantly audit an X (Twitter) profile. Use cases include verifying the legitimacy of viral accounts, auditing brand follower health, or investigating coordinated disinformation campaigns.

### 3.2 The Technology Stack
* **Frontend UI (Vanilla HTML/CSS/JS):** We strictly avoided bloated frameworks like React or Next.js. Utilizing Vanilla Web Components alongside CSS pseudo-elements ensured the UI remains completely monolithic, lightning-fast, and entirely devoid of complex Node package build steps. 
* **Backend Architecture (FastAPI - Python):** FastAPI handles the asynchronous rendering of the Jinja2 HTML templates while natively bridging the Python ML environment (Pandas/NumPy/XGBoost) securely.
* **Undocumented Scraper Protocol (`requests`):** To bypass the exorbitant costs and strict rate limits of the official X API, the platform utilizes a custom backend Python scraper targeting undocumented endpoints (`api.fxtwitter.com`). 

---

## 4. End-to-End Workflow: How it Operates

1. **User Request:** A user inputs an X handle (e.g., `@sundarpichai`).
2. **Metadata Bridging:** The backend `scrapper.py` quietly queries the external endpoint, completely bypassing local browser CORS restrictions, and retrieves raw profile JSON data.
3. **Mathematical Derivation:** The raw data (e.g., Joined in 2008) is converted into numerical ML features (e.g., `account_age_days = 6045`). Features are normalized via logarithmic scaling where necessary.
4. **Predictive Inference:** The Pandas array is fed sequentially into the `XGBoost` model for a probability score, and immediately shoved into the `.TreeExplainer` SHAP model for feature weight extraction. 
5. **Dashboard Generation:** FastAPI receives the data, calculates the visual SVG Gauge degrees, maps the SHAP Waterfall data, translates impacts into laymen's text, and auto-generates the customized UI canvas.

---

## 5. Future Prospects & Scoped Features

During the development cycle, several powerful features were conceptualized but sidelined to prioritize core functionality and immediate launch speeds.

### 5.1 Expandable Features
* **Recent Search History (`localStorage`):** Implementing a localized caching system in the user's browser, allowing them to quickly click previously analyzed profiles without re-fetching data.
* **Deep Text / NLP Analysis:** Scanning the user's last 50 tweets and funneling their text through a lightweight LLM (Local LLaMA) to detect repetitive, GPT-generated phrasing.
* **Share-to-X Verification Badges:** Allowing users to generate a "Certified Human/Bot" PNG passport card natively via canvas generation to post directly into ongoing Twitter threads.

### 5.2 Current Limitations & Roadblocks
* **Scraper Vulnerability:** The absolute biggest operational risk factor to MASKOFF is our reliance on the undocumented scraper. If X updates its architecture or explicitly rate-limits the backend server's IP address, the tool will temporarily break until the `scrapper.py` file is patched or an official Paid API tier is utilized.
* **NLP Speed Constraints:** NLP (Natural Language Processing) for tweet analysis was purposefully withheld. Deep text analysis would dramatically slow down the user experience, pushing our <2 second load times to localized speeds of 10+ seconds. 
* **Statelessness:** The app currently has no user authentication or rolling database, meaning we cannot conduct platform-wide macro-analysis / trend charting based on user search queries over time.

---
*Generated for the MASKOFF Production Finalization.*
