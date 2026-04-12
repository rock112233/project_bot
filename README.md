# MASKOFF Bot Detection 🎭🤖

MASKOFF is a highly accurate Machine Learning tool designed to identify hidden bot identities and fake accounts on Twitter (X) by evaluating live profile metrics. The system utilizes an XGBoost predictive model along with SHAP (SHapley Additive exPlanations) to provide explainable insights indicating *why* an account was flagged.

## 📂 Project Structure

This repository is strictly organized to separate code from datasets and documentation.

```
MASKOFF_Project/          # Core Application
├── src/                  
│   ├── main.py           # FastAPI Backend Server & ML routing
│   └── scrapper.py       # Twitter data extraction module
├── models/               
│   └── xgboost_model.pkl # Trained Machine Learning Model
├── config/               
│   └── feature_cols.json # Model Feature Maps
├── static/               
│   └── style.css         # UI Styling
└── templates/            
    └── index.html        # UI Frontend
```
> **Note**: Datasets, Jupyter Notebooks, Context Files, and Documentation are purposefully excluded from version control via `.gitignore` to maintain a clean production repository.

---

## 🚀 How to Run the Project Locally

Follow these step-by-step instructions to get the application running on your Windows machine.

### 1. Initial Setup & Virtual Environment

Start by cloning the project and setting up an isolated virtual environment using `uv` (a fast Python package installer and environment manager).

```powershell
# Step 1: Open your terminal inside the project root
# Step 2: Create a virtual environment
uv venv

# Step 3: Activate the virtual environment
.\.venv\Scripts\activate

# Step 4: Install dependencies
uv pip install fastapi uvicorn requests python-multipart jinja2 xgboost scikit-learn shap pandas numpy python-dotenv tweepy
```

### 2. Setting up the API `.env` File

*This project is capable of utilizing Official Twitter Developer API keys. Here is how to configure them securely:*

1. Go to the [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard).
2. Create a new "Project" and "App".
3. Navigate to **Keys and Tokens**.
4. Generate the following keys: **Consumer Key**, **Consumer Secret**, and **Bearer Token**.
5. Inside the root of your project directory, create a file named exactly `.env` (ensure it has no filename extension like `.txt`).
6. Open `.env` and paste your keys in this exact format:

```env
TWITTER_CONSUMER_KEY="your_extracted_api_key_here"
TWITTER_CONSUMER_SECRET="your_extracted_api_secret_here"
TWITTER_BEARER_TOKEN="your_extracted_bearer_token_here"
```
*(Your `.env` file is already listed in `.gitignore`, so these sensitive keys will never be accidentally pushed to GitHub).*

### 3. Starting the Application

With your environment active and `.env` configured, start the backend server:

```powershell
# Navigate into the project folder housing the source code
cd MASKOFF_Project

# Start the FastAPI web server
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### 4. Viewing the Web App
1. Open your web browser (Chrome/Edge/Brave).
2. Navigate to [http://localhost:8000](http://localhost:8000).
3. Enter a target Twitter handle (e.g., `elonmusk` or `sundarpichai`) into the Search Bar.
4. Click **Analyze** and wait for the AI to present the bot probability and its SHAP feature explanations!

---

## ⚠️ Things to Take Care Of

* **API Limits:** Free APIs and standard Developer endpoints have rate limits. If you spam the analyze button, Twitter might return a `429 Too Many Requests` or `402 Payment Required` block.
* **Environment Paths:** Always ensure you are running `uvicorn` while your `.venv` is actively initialized. If you get `ModuleNotFoundError`, it usually means you ran python natively outside the `.venv`.
* **Model Bias:** The model historically places heavy weight on the `verified` metric. Ensure you are evaluating public accounts. Private or suspended accounts will fail to scrape correctly.
