# 🎭 MASKOFF: Explainable Bot Detection Platform

MASKOFF is a high-performance web platform that utilizes Machine Learning and Explainable AI (XAI) to analyze Twitter/X profiles and predict the probability of an account being a bot. 

## 🚀 Newly Added Updates & Features

- **Value-Based Sentiment Explanations:** The AI reasoning has been upgraded with value-based sentiment logic, giving users crystal clear, human-readable explanations about *why* a specific behavior triggered a flag (e.g., "Account lacks a description, raising bot suspicion").
- **Live Insight Typewriter Animation:** The final AI verdict is now delivered through a sleek, delayed typewriter animation, providing a dynamic "AI thinking" aesthetic when analyzing profiles.
- **Interactive Profile Cards:** Usernames displayed in the profile block are now hyperlinked, allowing you to instantly jump to the live X.com profile of the analyzed account.
- **Premium Minimalist UI Overhaul:** Swapped the aggressive neon accents for a hyper-professional Slate & Blue design system with a completely refactored 2x2 responsive Grid Layout. 
- **SHAP Waterfall Component:** Implemented a bidirectional waterfall impact chart showing the exact numerical weight of each behavioral tracking metric.
- **Backend Optimization:** Deeply optimized the FastAPI asynchronous inference pipeline, achieving zero-blocking I/O and blazing fast response times by removing payload overhead.

## 📂 Project Structure

This repository uses a clean, production-ready architecture layout:

```text
├── docs/                     -> Extensive project documentation 
├── MASKOFF_Project/          -> Core Application
│   ├── config/               -> Model configuration and JSON feature maps
│   ├── src/                  -> Application backend (FastAPI inference & custom webscraper)
│   ├── static/               -> Application frontend CSS (Light/Dark themes)
│   └── templates/            -> Application frontend HTML (Jinja2)
├── model_training/           -> Machine learning and pipeline generation
│   ├── notebooks/            -> Development Jupyter notebooks 
│   ├── grid_search.py        -> GPU-accelerated XGBoost Hyperparameter tuning
│   └── train_new.py          -> Production model training pipeline
├── .env                      
└── README.md
```

## ⚙️ How it Works

1. **Undocumented API Profiling:** `MASKOFF_Project/src/scrapper.py` utilizes undocumented endpoints to fetch profile data in real-time without relying on paid X API keys or slow Selenium drivers.
2. **Metadata Processing:** Raw account features (Follower ratios, activity per day, bio flags) are mathematically derived and structured into a Pandas array.
3. **Algorithmic Inference:** Data goes through our `grid_search` optimized XGBoost Classifier (achieving 92.58% accuracy). The `SHAP` framework simultaneously extracts the real-time contribution values of the individual metrics.
4. **Jinja Rendering:** FastAPI magically injects the resulting calculation context directly into our custom HTML frontend canvas.

## 💻 Running Locally

1. Initialize the virtual environment natively via terminal:
   ```shell
   uv venv
   .\.venv\Scripts\activate
   uv pip install -r requirements.txt # (Ensure dependencies align to the pipeline)
   ```
2. Navigate to the App directory and boot the backend server:
   ```shell
   cd MASKOFF_Project
   uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
   ```
3. Open `http://127.0.0.1:8000/` in your browser.
