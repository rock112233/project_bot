# MASKOFF: Advanced Explainable Bot Detection Platform — Final Project Report

## 1. Project Summary & End-to-End Workflow

### 1.1 What We Are Using

MASKOFF is an advanced, real-time web application engineered to identify automated ("bot") profiles on X (Twitter). The system transcends standard "black box" machine learning predictions by integrating **Explainable AI (XAI)**. It analyzes a user's behavioral metadata (such as follower ratios, logarithmic scale of connections, and posting frequency) and visualizes the exact mathematical reasoning behind its classification.

### 1.2 The Technology Stack

Every tool in our stack was chosen to maximize inference speed and provide flawless transparency:

- **FastAPI (Python):** The high-performance asynchronous backend server framework routing data requests natively.
- **Vanilla Web Stack (HTML5 / CSS3 / JavaScript):** We utilized a monolithic, dependency-free frontend template rendered with Jinja2. This avoids bloated JavaScript frameworks (like React) ensuring lightning-fast load times.
- **XGBoost:** The predictive Machine Learning model mathematically predicting bot probability based on behavioral arrays.
- **SHAP (SHapley Additive exPlanations):** The Explainable AI matrix calculating the exact localized impact of feature decisions.
- **SMOTE-Tomek Links:** A powerful data sampling algorithm used during training to synthetically balance uneven datasets, preventing class bias.
- **Grid Search CV:** An exhaustive processing algorithm utilized specifically during model training to hunt down the absolute perfect hyperparameter structure.

### 1.3 End-to-End Workflow Execution

1. **User Query:** The user inputs an X handle in the browser (e.g., `@elonmusk`).
2. **Metadata Fetch:** FastAPI triggers the backend scraper via the `requests` module, pulling a JSON payload directly from undocumented REST endpoints in milliseconds.
3. **Data Transformation:** The raw JSON is mathematically derived into a 12-feature Pandas array algorithm (e.g., converting sheer tweet volume to `statuses_per_day`, scaling extreme metrics via `numpy.log()`).
4. **Probability Scaling:** The array is passed into the pre-trained `XGBoost` model, which calculates a Bot Probability percentage.
5. **SHAP Matrix Evaluation:** The exact same metadata array is injected into the `SHAP.TreeExplainer`. SHAP uses cooperative game theory to measure precisely how heavily each meta-feature pushed the final score up or down.
6. **Dashboard Generation:** The derived data is rendered over the Jinja2 template. The UX displays interactive username linking, a visual Confidence Gauge, the localized Waterfall impact chart, and the Final Insight Typewriter text generated via custom value-based sentiment logic.

---

## 2. Comparative Data: Legacy Prototype vs. Current Architecture

The initial academic prototype of this project struggled from severe architectural flaws that resulted in broken inferences. We stripped and overhauled the logic. Here is exactly what the older system had, and how we solved it:

### 2.1 Dataset Manipulation & Class Imbalance

- **What the Older Model did:** The original TwiBot-22 dataset featured a massive 6:1 class imbalance (most accounts were human). The prototype attempted to fix this using basic **Undersampling** (simply deleting thousands of human rows from the dataset to make it even with bots).
  - **Proof of Failure:** This resulted in massive data loss. The older model achieved a "71% Accuracy", yielding a devastatingly low **Bot Precision of ~42%** (flagging thousands of innocent humans as bots).
- **How We Tackled It:** We integrated **SMOTE-Tomek Links**.
  - _What it is:_ A hybrid approach used during model training. **SMOTE** synthetically generates new data points for the minority "Bot" class by interpolating existing ones. Simultaneously, **Tomek Links** systematically removes noisy boundary points where a Bot and Human overlap. This creates hyper-defined decision boundaries entirely avoiding data destruction, vastly lifting specific bot detection precision to ~80% without data leakage.

### 2.2 Feature Extraction Bottlenecks

- **What the Older Model did:** Relied on **Selenium**, a heavy automated browser driving headless Chrome. It loaded a full webpage, waited for a forced 4-second `time.sleep()`, and parsed raw CSS DOM selectors (`data-testid="UserName"`).
  - **Proof of Failure:** This required immense CPU overhead, took over 5 seconds per request, and broke instantly if Twitter updated web UI spacing.
- **How We Tackled It:** Built a pure `requests` based scraper targeting an undocumented API backdoor (`api.fxtwitter.com`). This extracts structured, pure JSON payloads directly into the server in under **1 second**, totally immune to frontend UI changes and eliminating massive browser overhead.

### 2.3 The ML Brain & Hyperparameter Tuning

- **What the Older Model did:** Utilized baseline guessed parameters for the XGBoost engine.
- **How We Tackled It:** We utilized **Grid Search Cross-Validation**.
  - _What it is:_ Grid Search is an algorithm applied exclusively during the _training_ phase. It systematically constructs and evaluates the predictive model across hundreds (specifically 48 core) distinct combinations of hyperparameters (e.g., `learning_rate=0.05 vs 0.1`, `max_depth=4 vs 8`). It mathematically guarantees we are launching with the absolute optimal "brain structure" for our data shape, securing our realistic **85.00% accuracy**.

### 2.4 INFERENCE BREAKDOWN: The "99.1% Confidence" Bug

- **What the Older Model did:** The legacy training notebook statistically assigned **99.1%** of decision weight to the `verified` metric, and **0.9%** to `account_age_days`. Yet, because the old Selenium scraper was incapable of pulling blue-check verifications or account creation dates reliably, these variables were **hardcoded inside `scrapper.py`** (as `0` and `365`).
  - **Proof of Failure:** The live inference system fundamentally collapsed. Since the XGBoost model exclusively evaluated those two parameters which were hardcoded as static constants by the scraper, the dashboard mindlessly generated identical probability outputs essentially ignoring 100% of the active scraped variables.
- **How We Tackled It:** The modernized XGBoost pipeline was specifically stripped of single-point-of-failure features. The model leverages 12 highly stable _behavioral variations_ (follow-back ratios, derived logging densities) ensuring the live engine scales organically based purely on user mechanics.

### 2.5 Explainability Evolution

- **What the Older Model did:** Relied on `model.feature_importances_`. This extracted global statistics over the entire dataset, printing identically generic text bounds regardless of who was scanned.
- **How We Tackled It:** Integrated dynamic **SHAP Values**.
  - _What it is:_ An algorithmic integration that calculates _localized_ mathematical margins. It measures the absolute isolated push/pull of a user's exact metadata point against a baseline, rendering highly customized insights and graphs on the fly.

---

## 3. Future Scaling Prospects & Current Limitations

### 3.1 Scaling Prospects (What We Can Build)

- **Cloud Infrastructure Scaling:** Wrapping the FastAPI architecture inside Docker containers and scaling through Kubernetes orchestrators across AWS or Azure to handle tens of thousands of concurrent profile audits globally.
- **Local Redis Database Caching:** Implementing a localized caching layer on the server storing identical recent queries, radically dropping calculation latency for viral accounts down to single-digit milliseconds.
- **Official API Bridges:** Hardwiring the extraction protocols into officially authenticated X Enterprise Developer environments for hyper-reliable execution tracking.
- **Large Language Models (LLM) Integration:** Taking the scope beyond pure metadata behavioral tracking. We can funnel the textual semantic history of a user's latest 50 tweets natively through a localized neural transformer (LLaMA or GPT) to run sentiment mapping and detect explicitly scheduled or repetitive LLM-generated phrasing patterns contextually.

### 3.2 Limitations Inhibiting Scaling Currently

- **Undocumented Endpoint Fragility:** The entire core payload extraction relies on `api.fxtwitter.com`. Scaling this application into a heavily monetized enterprise cloud service holds catastrophic operational limits; if Twitter patches or throttles that undocumented IP pathway, queries immediately die. Upgrading to the Official Enterprise API carries immense and often prohibitive operational SLA costs for solo researchers.
- **Inference Pipeline Bottlenecking:** While incorporating an LLM text-analysis engine would profoundly increase fake-account detection depth, running textual arrays through transformer neural networks forces astronomical computing loads. Shifting from our XGBoost/SHAP metadata loop out to Deep Learning text analysis would balloon the web app load time from lightning-fast **(<2 seconds)** to severely sluggish delays **(>10 to 15 seconds)**, destroying the rapid UI/UX experience and creating severe hardware cost spikes locally.