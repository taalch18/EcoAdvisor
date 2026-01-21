import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data and tuned ensemble model
esg_data = pd.read_csv(os.path.join(OUTPUT_DIR, "master_combined.csv"))
yahoo_data = pd.read_csv(os.path.join(OUTPUT_DIR, "real_esg_stock_data.csv"))

# Debug merge (keep for now, but can remove later)
print(f"Yahoo columns: {yahoo_data.columns.tolist()}")
print(f"ESG columns: {esg_data.columns.tolist()}")

# Fix merge: Yahoo 'CompanyID' is actually the ticker, rename to 'Ticker' if needed
if 'CompanyID' in yahoo_data.columns:
    yahoo_data.rename(columns={'CompanyID': 'Ticker'}, inplace=True)
    print("Renamed Yahoo 'CompanyID' to 'Ticker'")
else:
    print("Warning: 'CompanyID' not found in yahoo_data")

# Force remove duplicate 'Ticker' columns
yahoo_data = yahoo_data.loc[:, ~yahoo_data.columns.duplicated()]
print(f"Yahoo columns after duplicate removal: {yahoo_data.columns.tolist()}")

esg_data['Ticker'] = esg_data['Ticker'].astype(str) if 'Ticker' in esg_data.columns else None
yahoo_data['Ticker'] = yahoo_data['Ticker'].astype(str)

# Manually add tickers to ESG data (assuming order matches Yahoo) - TODO: Improve in preprocessing to avoid this hack
tickers_list = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'XOM', 'NEE']
esg_data['Ticker'] = [tickers_list[i % 10] for i in range(len(esg_data))]
esg_data['Ticker'] = esg_data['Ticker'].astype(str)

# Safe debug for tickers
esg_tickers = esg_data['Ticker'].dropna().unique()[:5] if 'Ticker' in esg_data.columns else []
yahoo_series = yahoo_data['Ticker'] if 'Ticker' in yahoo_data.columns else pd.Series()
yahoo_tickers = yahoo_series.dropna().unique()[:5] if not yahoo_series.empty else []
print(f"ESG sample tickers: {esg_tickers}")
print(f"Yahoo sample tickers: {yahoo_tickers}")

ensemble_model = joblib.load(os.path.join(MODEL_DIR, "esg_forecast_ensemble_tuned.pkl"))

# Build docs AFTER merge and ticker addition
def build_documents(df):
    latest = df.groupby("CompanyID").tail(1) if "CompanyID" in df.columns else df.tail(10)  # Fallback if no CompanyID
    docs, meta = [], []
    for _, row in latest.iterrows():
        company_id = str(row.get("CompanyID", "Unknown"))
        ticker = str(row.get("Ticker", company_id))
        summary = (
            f"Ticker: {ticker}. CompanyID: {company_id}. "
            f"Industry: {row.get('Industry','Unknown')}. "
            f"ESG Overall: {row.get('ESG_Overall','NA')}. "
            f"GrowthRate: {row.get('GrowthRate','NA')}. "
            f"Financial sentiment mean: {row.get('fin_sent_mean',0):.2f}. "
            f"Twitter sentiment mean: {row.get('tw_sent_mean',0):.2f}. "
            f"MarketCap: {row.get('MarketCap','NA')}, Revenue: {row.get('Revenue','NA')}."
        )
        docs.append(summary)
        meta.append(company_id)
    return docs, meta

docs, meta = build_documents(esg_data)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(docs).toarray()

# Retrieval
def retrieve_context(query, k=3):
    q_vec = vectorizer.transform([query]).toarray()
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    hits = [{"company_id": meta[idx], "score": float(sims[idx]), "text": docs[idx]} for idx in top_idx]
    return hits

# FIXED: Prediction with exact 15 features (fill missing with 0)
def predict_growth(company_id):
    # Get data for the company (try Ticker first, then CompanyID)
    ticker_data = esg_data[esg_data["Ticker"].astype(str).str.upper() == str(company_id).upper()]
    if ticker_data.empty:
        ticker_data = esg_data[esg_data["CompanyID"].astype(str) == str(company_id)]
    
    if ticker_data.empty:
        print(f"No data found for company: {company_id}")
        return None
    
    # Exact features the model expects (from train_ml.py)
    required_features = [
        "Revenue", "ProfitMargin", "MarketCap", "ESG_Overall", "ESG_Environmental",
        "ESG_Social", "ESG_Governance", "CarbonEmissions", "WaterUsage", "EnergyConsumption",
        "fin_sent_mean", "tw_sent_mean", "Revenue_lag1", "MarketCap_lag1", "ESG_Overall_lag1"
    ]
    
    # Take the latest row and build a DataFrame with all features, fill missing with 0
    sample_row = ticker_data.iloc[-1:].copy()
    sample_dict = {}
    for feat in required_features:
        if feat in sample_row.columns:
            sample_dict[feat] = sample_row[feat].iloc[0]
        else:
            sample_dict[feat] = 0.0  # Fill missing with 0
    
    sample_df = pd.DataFrame([sample_dict])
    
    try:
        pred = float(ensemble_model.predict(sample_df)[0])
        return pred
    except Exception as e:
        print(f"Prediction error for {company_id}: {e} (Feature shape: {sample_df.shape})")
        return None

# Company ID extraction (improved regex for tickers)
def extract_company_id(question):
    tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'XOM', 'NEE']
    for ticker in tickers:
        if ticker.upper() in question.upper():
            return ticker
    # Fallback: Extract any uppercase word (e.g., ticker)
    match = re.search(r'\b[A-Z]{1,5}\b', question.upper())
    return match.group(0) if match else None

# GPT-2 with FIXED: Shorter, directive prompt and better handling
generator = pipeline('text-generation', model='gpt2', max_new_tokens=30, truncation=True, pad_token_id=50256)  # Shorter output

# Updated RAG
def ask_ecowise(question):
    company_id = extract_company_id(question)
    hits = retrieve_context(question, k=3)
    pred = predict_growth(company_id) if company_id else None

    # Shorten context even more
    context_text = " ".join([h['text'][:100] for h in hits[:2]])
    pred_text = f"Predicted growth: {pred:.2f}%" if isinstance(pred, (int, float)) else "No prediction available."

    # FIXED: Directive prompt to avoid echoing
    prompt = f"Give a one-sentence ESG investment recommendation for {company_id or 'the company'} based on: {context_text}. {pred_text}"

    try:
        generated = generator(prompt, num_return_sequences=1)[0]['generated_text']
        # Extract only the new part after the prompt
        response = generated[len(prompt):].strip().split('.')[0] + '.'  # Limit to one sentence
        if not response or len(response) < 10:
            response = "HOLD (Neutral ESG, no strong signal)."
    except Exception as e:
        print(f"GPT-2 error: {e}")
        response = "HOLD (Generation failed)."

    full_response = []
    full_response.append("EcoWise Advisor (Ensemble ML + GPT-2 RAG)")
    full_response.append("-" * 70)
    full_response.append(f"Question: {question}")
    full_response.append(f"Detected Company ID: {company_id if company_id else 'None'}")
    full_response.append("\nTop Retrieved Context:")
    for i, h in enumerate(hits, 1):
        full_response.append(f"\n[{i}] Similarity Score: {h['score']:.3f} | CompanyID: {h['company_id']}")
        full_response.append(h["text"])
    full_response.append(f"\nEnsemble ML Growth Prediction: {pred_text}")
    full_response.append(f"\nGPT-2 Recommendation: {response}")
    
    return "\n".join(full_response)

# Run
if __name__ == "__main__":
    print("\nEcoWise Advisor Ready ✅ (Ensemble ML + GPT-2 RAG)")
    q = input("Enter your ESG query: ")
    print("\n" + ask_ecowise(q))
