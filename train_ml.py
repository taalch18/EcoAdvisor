import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)


def log(message: str) -> None:
    print(message)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

master_path = os.path.join(OUTPUT_DIR, "master_combined.csv")


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Master dataset not found at: {path}\n"
            f"Make sure preprocessing generated 'master_combined.csv' in outputs/."
        )
    df = pd.read_csv(path)
    log(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def add_lag_features(df: pd.DataFrame, lag_cols: list[str], group_col: str) -> pd.DataFrame:
    df = df.copy()
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby(group_col)[col].shift(1)
    return df


def main():
    df = load_dataset(master_path)

    target = "GrowthRate"
    features = [
        "Revenue",
        "ProfitMargin",
        "MarketCap",
        "ESG_Overall",
        "ESG_Environmental",
        "ESG_Social",
        "ESG_Governance",
        "CarbonEmissions",
        "WaterUsage",
        "EnergyConsumption",
        "fin_sent_mean",
        "tw_sent_mean",
    ]

    # One-hot encode categorical columns
    cat_cols = ["Industry", "Region", "ESG_Bucket"]
    present_cat_cols = [c for c in cat_cols if c in df.columns]
    if present_cat_cols:
        df = pd.get_dummies(df, columns=present_cat_cols, drop_first=True)

    # Add lag features for time series signal
    lag_cols = ["Revenue", "MarketCap", "ESG_Overall"]
    df = add_lag_features(df, lag_cols=lag_cols, group_col="CompanyID")

    lag_feature_names = [f"{col}_lag1" for col in lag_cols]
    features = features + lag_feature_names

    # Drop rows where lags or target are missing (first year per company will be missing lag)
    needed_cols = lag_feature_names + [target]
    df = df.dropna(subset=[c for c in needed_cols if c in df.columns])

    # Remove missing feature columns safely (in case master_combined is incomplete)
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master dataset: {missing}")

    X = df[features]
    y = df[target]

    log(f"Training data: {len(df)} rows, {X.shape[1]} features")

    # Hyperparameter tuning: HistGradientBoosting
    hgb_param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "max_iter": [400, 500],
    }

    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb_grid = GridSearchCV(hgb, hgb_param_grid, cv=3, scoring="r2")
    hgb_grid.fit(X, y)

    best_hgb = hgb_grid.best_estimator_
    log(f"Best HGB params: {hgb_grid.best_params_}, R²: {hgb_grid.best_score_:.4f}")

    # Hyperparameter tuning: RandomForest
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, None],
    }

    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring="r2")
    rf_grid.fit(X, y)

    best_rf = rf_grid.best_estimator_
    log(f"Best RF params: {rf_grid.best_params_}, R²: {rf_grid.best_score_:.4f}")

    # Ensemble model: weighted average of both models
    ensemble = VotingRegressor(
        estimators=[("hgb", best_hgb), ("rf", best_rf)],
        weights=[0.7, 0.3],
    )

    log("Starting TimeSeriesSplit cross-validation...")

    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []
    mse_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        log(f"Fold {fold}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mse_scores.append(mse)
        r2_scores.append(r2)

        log(f"  R²={r2:.4f}, MSE={mse:.4f}")

    log(f"\nMean R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores) * 2:.4f})")
    log(f"Mean MSE: {np.mean(mse_scores):.4f}")

    # Train final ensemble on full dataset
    ensemble.fit(X, y)

    model_path = os.path.join(MODEL_DIR, "esg_forecast_ensemble_tuned.pkl")
    joblib.dump(ensemble, model_path)
    log(f"\nSaved tuned ensemble model -> {model_path}")

    # Sample prediction
    sample = X.iloc[-1:]
    pred = ensemble.predict(sample)[0]
    actual = y.iloc[-1]

    log(f"Sample prediction: {pred:.2f} (Actual: {actual:.2f})")


if __name__ == "__main__":
    main()
