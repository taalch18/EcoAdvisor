import os
import re
import argparse
import numpy as np
import pandas as pd
import yfinance as yf


def log(message: str, level: str = "INFO") -> None:
    print(f"[{level}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="EcoWise Advisor - Data Preprocessing Pipeline")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to input data directory (default: <script_dir>/data)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Path to output directory (default: <script_dir>/outputs)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2019-01-01",
        help="Start date for Yahoo Finance history (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="End date for Yahoo Finance history (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,TSLA,GOOGL,AMZN,NVDA,JPM,V,XOM,NEE",
        help="Comma-separated list of tickers to fetch",
    )

    return parser.parse_args()


def fetch_yfinance_data(output_dir: str, start: str, end: str, tickers: list[str]):
    log("Fetching Yahoo Finance data...")

    frames = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)

            if hist.empty:
                log(f"{ticker}: no rows returned by yfinance", "WARN")
                continue

            info = getattr(stock, "info", {}) or {}

            hist = hist.copy()
            hist["Ticker"] = ticker
            hist["Industry"] = info.get("industry", "Unknown")
            hist["MarketCap"] = info.get("marketCap", np.nan)
            hist["Revenue"] = info.get("totalRevenue", np.nan)

            hist.reset_index(inplace=True)

            frames.append(hist)
            log(f"{ticker}: fetched {len(hist)} rows")

        except Exception as e:
            log(f"{ticker}: fetch failed ({e})", "ERROR")

    if not frames:
        # If fetch fails (e.g. no internet), create dummy data for structure if it doesn't exist
        log("No data fetched. Checking if file exists...", "WARN")
        out_path = os.path.join(output_dir, "real_esg_stock_data.csv")
        if os.path.exists(out_path):
            return pd.read_csv(out_path)
        else:
            # create dummy
            log("Creating dummy real_esg_stock_data.csv due to fetch failure", "WARN")
            df = pd.DataFrame({
                "Ticker": tickers,
                "Industry": ["Technology" if t in ["AAPL", "MSFT", "GOOGL", "NVDA"] else "Retail" if t == "AMZN" else "Automotive" if t == "TSLA" else "Financial Services" if t in ["JPM", "V"] else "Energy" for t in tickers],
                "MarketCap": [1e9] * len(tickers),
                "Revenue": [1e8] * len(tickers)
            })
            df.to_csv(out_path, index=False)
            return df

    df = pd.concat(frames, ignore_index=True)

    df["Year"] = df["Date"].dt.year
    df["CompanyID"] = df["Ticker"]

    # We don't need random ESG scores here anymore, we will get them from synthetic data
    # but keeping it for compatibility if needed elsewhere
    esg_scores = np.random.uniform(40, 80, len(df))
    df["ESG_Overall"] = esg_scores

    close_series = df.groupby("Ticker")["Close"]
    df["GrowthRate"] = close_series.pct_change(252) * 100

    out_path = os.path.join(output_dir, "real_esg_stock_data.csv")
    df.to_csv(out_path, index=False)
    log(f"Saved Yahoo Finance dataset: {len(df)} rows -> {out_path}")

    return df


def load_initial_data(data_dir: str):
    log("Loading initial datasets...")

    if not os.path.exists(data_dir):
        log(f"DATA_DIR not found: {data_dir}", "ERROR")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    available_files = os.listdir(data_dir)
    log(f"Available files in DATA_DIR: {available_files}")

    file_map = {
        "esg": "esg_data.csv",
        "fin": "sentiment_data.csv",
        "tw": "twitter_sentiment.csv",
    }

    esg = pd.DataFrame()
    fin_sent = pd.DataFrame()
    tw_sent = pd.DataFrame()

    esg_path = os.path.join(data_dir, file_map["esg"])
    fin_path = os.path.join(data_dir, file_map["fin"])
    tw_path = os.path.join(data_dir, file_map["tw"])

    if os.path.exists(esg_path):
        esg = pd.read_csv(esg_path)
        log(f"ESG loaded: {len(esg)} rows")
    else:
        log(f"Missing ESG file: {file_map['esg']}", "WARN")

    if os.path.exists(fin_path):
        fin_sent = pd.read_csv(fin_path)
        log(f"Financial sentiment loaded: {len(fin_sent)} rows")
    else:
        log(f"Missing financial sentiment file: {file_map['fin']}", "WARN")

    if os.path.exists(tw_path):
        tw_sent = pd.read_csv(tw_path)
        log(f"Twitter sentiment loaded: {len(tw_sent)} rows")
    else:
        log(f"Missing twitter sentiment file: {file_map['tw']}", "WARN")

    return esg, fin_sent, tw_sent

def align_tickers_to_esg(esg_raw, real_stock_data):
    """
    Maps real Tickers to synthetic ESG data based on Industry.
    """
    log("Aligning Tickers to Synthetic ESG Data...")

    if esg_raw.empty or real_stock_data.empty:
        log("ESG or Real Stock data is empty, skipping alignment.", "WARN")
        return esg_raw

    # Get unique Tickers and their Industries from real data
    if "Ticker" not in real_stock_data.columns:
        log("Ticker column missing in real stock data.", "ERROR")
        return esg_raw

    # Assume Industry is in real_stock_data. If not, we can't map effectively.
    if "Industry" not in real_stock_data.columns:
        # Fallback: Assign tickers round-robin if Industry is missing
        tickers = real_stock_data["Ticker"].unique()
        esg_raw["Ticker"] = [tickers[i % len(tickers)] for i in range(len(esg_raw))]
        log("Industry missing in real stock data. Assigned tickers round-robin.", "WARN")
        return esg_raw

    ticker_info = real_stock_data[["Ticker", "Industry"]].drop_duplicates()

    # Clean industries for better matching (lowercase, simple match)
    def clean_ind(x): return str(x).lower().strip()
    ticker_info["Industry_Clean"] = ticker_info["Industry"].apply(clean_ind)
    esg_raw["Industry_Clean"] = esg_raw["Industry"].apply(clean_ind)

    aligned_frames = []
    synthetic_companies = esg_raw[["CompanyID", "Industry_Clean"]].drop_duplicates()

    # Manual mapping for known yahoo industries to synthetic industries
    industry_map = {
        "consumer electronics": "technology",
        "software - infrastructure": "technology",
        "internet content & information": "technology",
        "semiconductors": "technology",
        "information technology services": "technology",
        "internet retail": "retail",
        "specialty retail": "retail",
        "auto manufacturers": "transportation", # or manufacturing
        "banks - diversified": "finance",
        "credit services": "finance",
        "capital markets": "finance",
        "financial data & stock exchanges": "finance",
        "insurance - diversified": "finance",
        "oil & gas integrated": "energy",
        "utilities - regulated electric": "utilities",
        "utilities - renewable": "utilities"
    }

    for _, row in ticker_info.iterrows():
        ticker = row["Ticker"]
        real_ind = row["Industry_Clean"]

        # Try to map real industry to synthetic industry
        target_ind = industry_map.get(real_ind)

        # If not in map, try keyword matching
        if not target_ind:
            if "technology" in real_ind or "software" in real_ind or "semiconductor" in real_ind:
                target_ind = "technology"
            elif "retail" in real_ind:
                target_ind = "retail"
            elif "bank" in real_ind or "finance" in real_ind or "credit" in real_ind or "capital" in real_ind:
                target_ind = "finance"
            elif "energy" in real_ind or "oil" in real_ind or "gas" in real_ind:
                target_ind = "energy"
            elif "utility" in real_ind or "electric" in real_ind:
                target_ind = "utilities"
            elif "health" in real_ind or "drug" in real_ind or "biotech" in real_ind:
                target_ind = "healthcare"
            elif "auto" in real_ind or "transport" in real_ind or "airline" in real_ind:
                target_ind = "transportation"
            elif "manufacturing" in real_ind or "industrial" in real_ind:
                target_ind = "manufacturing"
            else:
                target_ind = real_ind # try exact match

        # Find matching synthetic companies
        matches = synthetic_companies[synthetic_companies["Industry_Clean"] == target_ind]

        if matches.empty:
             # Fallback: pick any random company
            matches = synthetic_companies
            log(f"No match for {ticker} (Real: {real_ind} -> Target: {target_ind}), using random fallback.", "WARN")
        else:
             log(f"Matched {ticker} (Real: {real_ind}) -> Target: {target_ind}")

        # Pick one deterministically based on ticker hash to be consistent
        seed = int(sum(ord(c) for c in ticker))
        chosen_id = matches.sample(1, random_state=seed).iloc[0]["CompanyID"]

        # Get that company's data
        company_data = esg_raw[esg_raw["CompanyID"] == chosen_id].copy()

        # Replace info
        company_data["Ticker"] = ticker
        company_data["Original_CompanyID"] = chosen_id
        # We replace Industry with real industry just in case
        company_data["Industry"] = row["Industry"]

        aligned_frames.append(company_data)

    if not aligned_frames:
        return esg_raw

    aligned_esg = pd.concat(aligned_frames, ignore_index=True)

    # Drop temp col
    aligned_esg.drop(columns=["Industry_Clean"], inplace=True)
    esg_raw.drop(columns=["Industry_Clean"], inplace=True) # clean up original too

    log(f"Aligned ESG data: {len(aligned_esg)} rows for {len(ticker_info)} tickers.")
    return aligned_esg


def preprocess_data(esg, fin_sent, tw_sent):
    log("Preprocessing datasets...")

    if not esg.empty:
        if "ESG_Overall" in esg.columns:
            esg = esg.dropna(subset=["ESG_Overall"])

        if "Year" not in esg.columns and "Date" in esg.columns:
            esg["Date"] = pd.to_datetime(esg["Date"], errors="coerce")
            esg["Year"] = esg["Date"].dt.year

        if "ESG_Overall" in esg.columns:
            bins = [0, 40, 70, 100]
            labels = ["Low", "Medium", "High"]
            esg["ESG_Bucket"] = pd.cut(esg["ESG_Overall"], bins=bins, labels=labels)

    if not fin_sent.empty:
        fin_sent.columns = fin_sent.columns.str.strip()
        # ... (Same cleaning as before)
        if "Sentence" in fin_sent.columns:
            sentences = fin_sent["Sentence"].astype(str).str.lower()
            sentences = sentences.str.replace(r"[^\w\s]", "", regex=True)
            fin_sent["Sentence"] = sentences

        if "Sentiment" in fin_sent.columns:
            raw = fin_sent["Sentiment"].astype(str).str.lower()
            mapped = raw.map({"positive": 1, "negative": -1, "neutral": 0})
            fin_sent["Sentiment"] = mapped.fillna(0).astype(int)

        if "Year" not in fin_sent.columns:
             # ...
             fin_sent["Year"] = 2023 # Default

    if not tw_sent.empty:
        tw_sent.columns = tw_sent.columns.str.strip()
        # ...
        if "Sentiment" in tw_sent.columns:
            tw_sent["Sentiment"] = pd.to_numeric(tw_sent["Sentiment"], errors="coerce").fillna(0).astype(int)

        if "Year" not in tw_sent.columns:
            tw_sent["Year"] = 2023

    log("Preprocessing complete.")
    return esg, fin_sent, tw_sent


def combine_data(esg, fin_sent, tw_sent):
    log("Combining datasets...")

    master = esg.copy() if not esg.empty else pd.DataFrame()

    if not fin_sent.empty:
        fin_by_year = fin_sent.groupby("Year", as_index=False)["Sentiment"].mean()
        fin_by_year.rename(columns={"Sentiment": "fin_sent_mean"}, inplace=True)

        if master.empty:
            master = fin_by_year
        else:
            master = master.merge(fin_by_year, on="Year", how="left")

    if not tw_sent.empty:
        tw_by_year = tw_sent.groupby("Year", as_index=False)["Sentiment"].mean()
        tw_by_year.rename(columns={"Sentiment": "tw_sent_mean"}, inplace=True)

        if master.empty:
            master = tw_by_year
        else:
            master = master.merge(tw_by_year, on="Year", how="left")

    if "fin_sent_mean" not in master.columns:
        master["fin_sent_mean"] = 0
    if "tw_sent_mean" not in master.columns:
        master["tw_sent_mean"] = 0

    master["fin_sent_mean"] = master["fin_sent_mean"].fillna(0)
    master["tw_sent_mean"] = master["tw_sent_mean"].fillna(0)

    # Add Lag Features
    log("Adding Lag Features...")
    if "Ticker" in master.columns and "Year" in master.columns:
        master.sort_values(by=["Ticker", "Year"], inplace=True)
        lag_cols = ["Revenue", "MarketCap", "ESG_Overall"]
        for col in lag_cols:
            if col in master.columns:
                master[f"{col}_lag1"] = master.groupby("Ticker")[col].shift(1)
            else:
                master[f"{col}_lag1"] = 0

        # Fill NaN lags with 0 (for first year)
        for col in lag_cols:
            master[f"{col}_lag1"] = master[f"{col}_lag1"].fillna(0)
    else:
        log("Ticker or Year column missing, cannot add lags properly.", "WARN")

    log(f"Master dataset ready: {len(master)} rows")
    return master


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = args.data_dir if args.data_dir else os.path.join(base_dir, "data")
    output_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    log(f"DATA_DIR: {data_dir}")
    log(f"OUTPUT_DIR: {output_dir}")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    # 1. Fetch/Get Real Stock Data
    real_stock_data = fetch_yfinance_data(output_dir, args.start, args.end, tickers)

    # 2. Load Initial Data
    esg_raw, fin_raw, tw_raw = load_initial_data(data_dir)

    # 3. Align Tickers
    esg_aligned = align_tickers_to_esg(esg_raw, real_stock_data)

    esg_clean = esg_aligned.copy()
    fin_clean = fin_raw.copy()
    tw_clean = tw_raw.copy()

    # 4. Preprocess
    esg_clean, fin_clean, tw_clean = preprocess_data(esg_clean, fin_clean, tw_clean)

    # 5. Combine and Add Lags
    master = combine_data(esg_clean, fin_clean, tw_clean)

    # Save
    master_path = os.path.join(output_dir, "master_combined.csv")
    master.to_csv(master_path, index=False)

    log(f"Saved master dataset -> {master_path}")
    log(f"Final shape: {master.shape}")
    log(f"Columns: {master.columns.tolist()}")
    log("Sample preview:")
    print(master.head(3))


if __name__ == "__main__":
    main()
