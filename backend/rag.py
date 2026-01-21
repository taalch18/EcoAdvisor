import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

class EcoAdvisor:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "outputs")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")

        print("Loading data...")
        self.esg_data = self._load_data()
        print("Loading model...")
        self.ensemble_model = self._load_model()

        print("Building RAG index...")
        self.vectorizer, self.tfidf_matrix, self.docs, self.meta = self._build_rag_index()
        print("Loading GPT-2...")
        self.generator = pipeline('text-generation', model='gpt2', max_new_tokens=60, truncation=True, pad_token_id=50256)

    def _load_data(self):
        path = os.path.join(self.OUTPUT_DIR, "master_combined.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}")
        df = pd.read_csv(path)
        # Ensure Ticker column exists and is string
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str)
        else:
             df['Ticker'] = "Unknown"
        return df

    def _load_model(self):
        path = os.path.join(self.MODEL_DIR, "esg_forecast_ensemble_tuned.pkl")
        if not os.path.exists(path):
             raise FileNotFoundError(f"Model not found at {path}")
        return joblib.load(path)

    def _build_rag_index(self):
        # Build documents
        # Group by Ticker if available, else CompanyID
        group_col = "Ticker" if "Ticker" in self.esg_data.columns else "CompanyID"
        latest = self.esg_data.groupby(group_col).tail(1)

        docs, meta = [], []
        for _, row in latest.iterrows():
            ticker = str(row.get("Ticker", "Unknown"))
            company_id = str(row.get("CompanyID", "Unknown"))
            industry = row.get('Industry', 'Unknown')
            esg_overall = row.get('ESG_Overall', 'NA')
            growth = row.get('GrowthRate', 'NA')

            summary = (
                f"Ticker: {ticker}. Industry: {industry}. "
                f"ESG Overall: {esg_overall}. GrowthRate: {growth}. "
                f"Revenue: {row.get('Revenue', 'NA')}. "
                f"MarketCap: {row.get('MarketCap', 'NA')}."
            )
            docs.append(summary)
            meta.append(ticker) # Use Ticker as ID for retrieval matching if possible

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(docs).toarray()

        return vectorizer, tfidf_matrix, docs, meta

    def retrieve_context(self, query, k=3):
        q_vec = self.vectorizer.transform([query]).toarray()
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        hits = [{"ticker": self.meta[idx], "score": float(sims[idx]), "text": self.docs[idx]} for idx in top_idx]
        return hits

    def predict_growth(self, ticker):
        if not ticker:
            return None

        # Find data for ticker
        ticker_data = self.esg_data[self.esg_data["Ticker"].str.upper() == ticker.upper()]

        if ticker_data.empty:
            return None

        # Exact features needed
        required_features = [
            "Revenue", "ProfitMargin", "MarketCap", "ESG_Overall", "ESG_Environmental",
            "ESG_Social", "ESG_Governance", "CarbonEmissions", "WaterUsage", "EnergyConsumption",
            "fin_sent_mean", "tw_sent_mean", "Revenue_lag1", "MarketCap_lag1", "ESG_Overall_lag1"
        ]

        sample_row = ticker_data.iloc[-1:].copy()
        sample_dict = {}
        for feat in required_features:
            if feat in sample_row.columns:
                sample_dict[feat] = sample_row[feat].iloc[0]
            else:
                sample_dict[feat] = 0.0

        sample_df = pd.DataFrame([sample_dict])

        try:
            pred = float(self.ensemble_model.predict(sample_df)[0])
            return pred
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def extract_ticker(self, question):
        # Simple extraction based on known tickers or uppercase words
        # Prioritize known tickers from data
        known_tickers = self.esg_data['Ticker'].unique()
        for t in known_tickers:
            if t.upper() in question.upper():
                return t

        match = re.search(r'\b[A-Z]{1,5}\b', question.upper())
        return match.group(0) if match else None

    def ask(self, question):
        ticker = self.extract_ticker(question)
        hits = self.retrieve_context(question)
        pred = self.predict_growth(ticker)

        context_text = " ".join([h['text'] for h in hits[:2]])
        pred_text = f"{pred:.2f}%" if pred is not None else "Unknown"

        # Improved Prompt: Directive and shorter
        prompt = (
            f"Task: Provide an ESG investment recommendation for {ticker or 'the company'}.\n"
            f"Data: {context_text}\n"
            f"Prediction: Growth {pred_text}\n"
            f"Advice (BUY/HOLD/SELL) and reason:"
        )

        try:
            # max_new_tokens ensures we don't get too much repetition
            generated = self.generator(prompt, num_return_sequences=1)[0]['generated_text']
            # We want the text AFTER the prompt
            response = generated[len(prompt):].strip()

            # Simple cleanup: take the first sentence or until newline
            if "\n" in response:
                response = response.split("\n")[0]
            elif "." in response:
                # Take first 2 sentences max
                sentences = response.split(".")
                response = ".".join(sentences[:2]) + "."

            if not response:
                response = "HOLD. Insufficient data to form a strong opinion."

        except Exception as e:
            print(f"GPT-2 error: {e}")
            response = "HOLD. Error in advice generation."

        return {
            "question": question,
            "ticker": ticker,
            "prediction": pred,
            "recommendation": response,
            "context": hits
        }
