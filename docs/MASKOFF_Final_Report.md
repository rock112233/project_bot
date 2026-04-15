# MASKOFF: Explainable Bot Detection Platform — Final Project Report

## 1. Executive Summary
MASKOFF is an advanced, high-performance web platform utilizing Machine Learning (ML) and Explainable AI (XAI) to accurately detect automated bot profiles on Twitter (X). Designed with speed, precision, and transparency in mind, MASKOFF transitions the opaque "black box" nature of traditional machine learning into a user-friendly, visually transparent dashboard.

This document serves as the comprehensive architectural knowledge dump, contrasting the flawed legacy academic prototype with the highly-optimized production platform we have finalized today.

---

## 2. The Legacy Prototype: How It Used to Work & Why It Failed

The original academic prototype of this project struggled from severe architectural constraints and inference-level breakdowns. It was originally engineered upon the **TwiBot-22 Dataset** with the following flawed logic:

### 2.1 The Data Imbalance & Undersampling Error
The legacy dataset suffered from a 6:1 (Human to Bot) class imbalance. To attempt correction, the model utilized **Undersampling**. 
* **The Result:** The legacy XGBoost model reported a misleading `76% Accuracy`, but effectively accomplished this by blindly guessing "Human" for almost every profile. The actual Bot Recall was merely **11% to 50%**, meaning it consistently failed to identify at least half of the actual automated accounts.

### 2.2 The Selenium Scraping Bottleneck
The older application relied heavily on the `Selenium` web driver running a headless Chrome browser to scrape Twitter/X. 
* **The Result:** The application was painstakingly slow (forcing a mandatory 4-second blind `time.sleep()`) and fundamentally brittle, as it relied on CSS DOM targeting (`[data-testid="UserName"]`) which frequently broke whenever Twitter deployed a silent site update.

### 2.3 PROOF OF INFRASTRUCTURE FAILURE (The Hardcoded Inference Bug)
The most critical error in the legacy structure occurred natively within its ML inference bridge.
1. The original XGBoost training notebook mathematically consolidated almost all decision-making weight onto exactly two columns: `verified` (**99.1%**) and `account_age_days` (**0.9%**). 
2. However, because the backend Selenium scraper was incapable of dynamically calculating account creation dates or scraping blue-check API data, these variables were **hardcoded inside `scrapper.py`** (e.g., `"verified": 0`, `"account_age_days": 365`).
* **The Result:** The live inference system was functionally broken. Because 100% of the mathematical weight was static, the model indiscriminately yielded nearly identical probability outputs for every single user analyzed. 

---

## 3. The Modernized Platform: What We Are Doing Better

We completely ripped out, restructured, and optimized the backend to achieve our current pipeline. 

### 3.1 Undocumented API Payload Extraction
* **The Fix:** We entirely abandoned Selenium. Instead, the current backend utilizes an undocumented, free REST endpoint (`api.fxtwitter.com`) via the Python `requests` library. 
* **The Benefit:** Data retrieval instantly avoids browser CORS restrictions, drops extraction times from >4 seconds to **under 1 second**, and flawlessly returns precise internal metrics (including exact verification statuses and creation timestamps) that unlock the algorithm's actual data weighting.

### 3.2 Hyperparameter XGBoost Optimization
* **The Fix:** The legacy model parameters were essentially baseline guesses. We mathematically locked our bounds by funneling the training array through an exhaustive **Grid Search** across 864 unique hyperparameter combinations (depth, learning rate, subsampling). 
* **The Benefit:** The production-ready XGBoost model natively digests nonlinear boundaries achieving a confirmed **92.58% accuracy**, without the gross false-negative rates of the previous undersampled prototype.

### 3.3 Dynamic SHAP (Explainable AI)
* **The Fix The Legacy "Black Box":** The old model utilized `model.feature_importances_`, outputting a *global* static explanation (literally generating identical text for every user). We integrated the **SHAP (SHapley Additive exPlanations)** framework.
* **The Benefit:** SHAP calculates exact local probabilities on the fly. This calculates the literal numerical margin of impact that feature had on that *specific* user, driving our "Value-Based Sentiment" UI to explain its logic elegantly in real-time.

---

## 4. Current Technology Stack & Use Case

### 4.1 The Use Case
MASKOFF allows security researchers, brand consultants, and social media auditors to immediately identify orchestrated disinformation campaigns or perform follower health audits without needing prohibitive Enterprise Twitter API keys.

### 4.2 The Technology Stack
* **Frontend UI (Vanilla HTML/CSS/JS):** Monolithic, lightning-fast rendering devoid of bloated React dependencies utilizing an elegant Light/Dark mode responsive grid.
* **Backend Architecture (FastAPI - Python):** Fully asynchronous pipeline handling Jinja2 template rendering while executing heavy ML arrays securely entirely server-side.
* **ML Engines:** Pandas, NumPy, XGBoost (Prediction), SHAP (Extractive XAI Interpretation).

---

## 5. End-to-End Workflow

1. **Query Input:** User hits `@username` in the UI.
2. **Metadata Bridging:** FastAPI pings `scrapper.py`, querying `fxtwitter` endpoints for JSON payloads, completely circumventing API paywalls.
3. **Data Pipeline:** 12 specific features (followers, following, account derivations) are normalized logarithmically and transformed into an active Pandas array.
4. **Predictive Inference:** The payload passes through XGBoost for localized scoring. Our `SHAP.TreeExplainer` maps the exact numerical variance.
5. **Dashboard Generation:** The system visually maps the data via a visual SVG Gauge, bidirectional SHAP Impact Waterfall Chart, and animates a dynamic "AI Typewriter" Insight Summary.

---

## 6. Future Prospects & Defined Limitations

### 6.1 Future Scalability
* **Localized Database Caching (`localStorage`):** We plan on implementing a short-term browser-level cache to hold "Recent Searches," saving bandwidth for repeated investigative queries.
* **Deep Neural NLP Analysis:** Currently, the system strictly looks at *behavioral metadata*. A future integration with a local LLM (like LLaMA) could evaluate the semantic linguistic structure of a user's recent tweets to detect AI-generated dialogue.

### 6.2 Application Roadblocks & Limitations
* **The Scraper Vulnerability:** Because MASKOFF avoids the catastrophic costs of official X/Twitter Enterprise Api plans, it fundamentally relies on an undocumented backdoor endpoint. If X abruptly shuts off this endpoint, the data pipeline halts entirely.
* **Why NLP Isn't Currently Embedded:** Adding Deep Text/Semantic analysis would drastically cripple the user experience. Shifting arrays into text-models would spike the backend inference from **<2 seconds** to **>10 seconds**. MASKOFF prioritizes speed over linguistic profiling.
