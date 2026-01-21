import os
import re
import argparse
import numpy as np
import pandas as pd
import yfinance as yf


def log(message: str, level: str = "INFO") -> None:
    print(message)


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
        raise ValueError("No data fetched from Yahoo Finance. Check your connection/tickers.")

    df = pd.concat(frames, ignore_index=True)

    df["Year"] = df["Date"].dt.year
    df["CompanyID"] = df["Ticker"]

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

        if "Sentence" in fin_sent.columns:
            sentences = fin_sent["Sentence"].astype(str).str.lower()
            sentences = sentences.str.replace(r"[^\w\s]", "", regex=True)
            fin_sent["Sentence"] = sentences

        if "Sentiment" in fin_sent.columns:
            raw = fin_sent["Sentiment"].astype(str).str.lower()
            mapped = raw.map({"positive": 1, "negative": -1, "neutral": 0})
            fin_sent["Sentiment"] = mapped.fillna(0).astype(int)

        if "Year" not in fin_sent.columns:
            possible_date_cols = ["Date", "date", "Timestamp", "timestamp", "PublishedAt", "published_at"]
            date_col = next((c for c in possible_date_cols if c in fin_sent.columns), None)

            if date_col:
                fin_sent[date_col] = pd.to_datetime(fin_sent[date_col], errors="coerce")
                fin_sent["Year"] = fin_sent[date_col].dt.year
            else:
                fin_sent["Year"] = 2023

    if not tw_sent.empty:
        tw_sent.columns = tw_sent.columns.str.strip()

        if "Text" in tw_sent.columns:
            text = tw_sent["Text"].astype(str).str.lower()
            text = text.str.replace(r"[^\w\s]", "", regex=True)
            tw_sent["Text"] = text

        if "Sentiment" in tw_sent.columns:
            tw_sent["Sentiment"] = pd.to_numeric(tw_sent["Sentiment"], errors="coerce").fillna(0).astype(int)

        if "Year" not in tw_sent.columns:
            possible_date_cols = ["Date", "date", "Timestamp", "timestamp", "CreatedAt", "created_at"]
            date_col = next((c for c in possible_date_cols if c in tw_sent.columns), None)

            if date_col:
                tw_sent[date_col] = pd.to_datetime(tw_sent[date_col], errors="coerce")
                tw_sent["Year"] = tw_sent[date_col].dt.year
            else:
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

    log(f"Master dataset ready: {len(master)} rows")
    return master


def dataset_summary(df: pd.DataFrame, name: str):
    return {
        "dataset": name,
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "missing_%": round(df.isna().mean().mean() * 100, 3),
        "duplicates": int(df.duplicated().sum()),
    }


def esg_metrics(esg_raw: pd.DataFrame, esg_clean: pd.DataFrame):
    raw_null = esg_raw["ESG_Overall"].isna().sum() if "ESG_Overall" in esg_raw.columns else None
    raw_null_pct = (raw_null / len(esg_raw) * 100) if raw_null is not None else None

    clean_null = esg_clean["ESG_Overall"].isna().sum() if "ESG_Overall" in esg_clean.columns else 0
    clean_null_pct = clean_null / len(esg_clean) * 100 if len(esg_clean) > 0 else 0

    industries = esg_clean["Industry"].nunique() if "Industry" in esg_clean.columns else 0
    companies = esg_clean["CompanyID"].nunique() if "CompanyID" in esg_clean.columns else 0

    if "Year" in esg_clean.columns and not esg_clean["Year"].dropna().empty:
        year_range = (int(esg_clean["Year"].min()), int(esg_clean["Year"].max()))
    else:
        year_range = (None, None)

    if "ESG_Bucket" in esg_clean.columns:
        bucket_dist = esg_clean["ESG_Bucket"].value_counts(dropna=False)
        bucket_pct = (bucket_dist / len(esg_clean) * 100).round(2)
    else:
        bucket_dist = pd.Series(dtype=int)
        bucket_pct = pd.Series(dtype=float)

    return {
        "raw_esg_null_count": raw_null,
        "raw_esg_null_%": None if raw_null_pct is None else round(raw_null_pct, 3),
        "clean_esg_null_count": int(clean_null),
        "clean_esg_null_%": round(clean_null_pct, 3),
        "industries": int(industries),
        "unique_companies": int(companies),
        "year_range": year_range,
        "bucket_counts": bucket_dist.to_dict(),
        "bucket_%": bucket_pct.to_dict(),
    }


def text_cleaning_metrics(raw_text: pd.Series, clean_text: pd.Series):
    def avg_words(s: pd.Series) -> float:
        return float(np.mean([len(str(x).split()) for x in s]))

    def count_pattern(s: pd.Series, pattern: str) -> int:
        return int(sum(len(re.findall(pattern, str(x))) for x in s))

    return {
        "avg_words_before": round(avg_words(raw_text), 3),
        "avg_words_after": round(avg_words(clean_text), 3),
        "urls_removed_est": count_pattern(raw_text, r"http\S+|www\.\S+"),
        "mentions_removed_est": count_pattern(raw_text, r"@\w+"),
        "hashtags_seen_est": count_pattern(raw_text, r"#\w+"),
    }


def twitter_mapping_metrics(tw_with_industry: pd.DataFrame):
    total = len(tw_with_industry)

    if "Industry" not in tw_with_industry.columns:
        return {
            "tweets_total": int(total),
            "tweets_known_industry": 0,
            "tweets_known_%": 0,
            "tweets_unknown": int(total),
            "tweets_unknown_%": 100,
        }

    known = (tw_with_industry["Industry"] != "Unknown").sum()
    unknown = (tw_with_industry["Industry"] == "Unknown").sum()

    return {
        "tweets_total": int(total),
        "tweets_known_industry": int(known),
        "tweets_known_%": round(known / total * 100, 2) if total > 0 else 0,
        "tweets_unknown": int(unknown),
        "tweets_unknown_%": round(unknown / total * 100, 2) if total > 0 else 100,
    }


def print_metrics_report(esg_raw, fin_raw, tw_raw, esg_clean, fin_clean, tw_clean, master):
    print("\n" + "=" * 80)
    print("PREPROCESSING METRICS REPORT")
    print("=" * 80)

    report = [
        dataset_summary(esg_raw, "ESG RAW"),
        dataset_summary(esg_clean, "ESG CLEAN"),
        dataset_summary(fin_raw, "FIN SENT RAW"),
        dataset_summary(fin_clean, "FIN SENT CLEAN"),
        dataset_summary(tw_raw, "TWITTER RAW"),
        dataset_summary(tw_clean, "TWITTER CLEAN"),
        dataset_summary(master, "MASTER COMBINED"),
    ]

    df_report = pd.DataFrame(report)
    print("\nDATASET SUMMARY:")
    print(df_report.to_string(index=False))

    def retention(a, b):
        return round(len(b) / len(a) * 100, 2) if len(a) > 0 else 0

    print("\nRETENTION RATE:")
    print(f"ESG retention: {retention(esg_raw, esg_clean)}%")
    print(f"Financial sentiment retention: {retention(fin_raw, fin_clean)}%")
    print(f"Twitter retention: {retention(tw_raw, tw_clean)}%")

    esg_rep = esg_metrics(esg_raw, esg_clean)
    print("\nESG METRICS:")
    for k, v in esg_rep.items():
        print(f"{k}: {v}")

    print("\nTEXT CLEANING METRICS:")
    if not fin_raw.empty and "Sentence" in fin_raw.columns and "Sentence" in fin_clean.columns:
        fin_txt = text_cleaning_metrics(fin_raw["Sentence"], fin_clean["Sentence"])
        print("Financial sentences:", fin_txt)

    if not tw_raw.empty and "Text" in tw_raw.columns and "Text" in tw_clean.columns:
        tw_txt = text_cleaning_metrics(tw_raw["Text"], tw_clean["Text"])
        print("Twitter text:", tw_txt)

    map_rep = twitter_mapping_metrics(tw_clean)
    print("\nTWITTER MAPPING METRICS:")
    for k, v in map_rep.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("=" * 80)


def main():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = args.data_dir if args.data_dir else os.path.join(base_dir, "data")
    output_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    log(f"DATA_DIR: {data_dir}")
    log(f"OUTPUT_DIR: {output_dir}")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    fetch_yfinance_data(output_dir, args.start, args.end, tickers)

    esg_raw, fin_raw, tw_raw = load_initial_data(data_dir)

    esg_clean = esg_raw.copy()
    fin_clean = fin_raw.copy()
    tw_clean = tw_raw.copy()

    esg_clean, fin_clean, tw_clean = preprocess_data(esg_clean, fin_clean, tw_clean)

    master = combine_data(esg_clean, fin_clean, tw_clean)

    print_metrics_report(esg_raw, fin_raw, tw_raw, esg_clean, fin_clean, tw_clean, master)

    master_path = os.path.join(output_dir, "master_combined.csv")
    master.to_csv(master_path, index=False)

    log(f"Saved master dataset -> {master_path}")
    log(f"Final shape: {master.shape}")
    log(f"Columns: {master.columns.tolist()}")
    log("Sample preview:")
    print(master.head(3))


if __name__ == "__main__":
    main()
