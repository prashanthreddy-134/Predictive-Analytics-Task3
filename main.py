from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "historical_sales.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["TimeIndex"] = np.arange(len(df))
    # Cyclic month features
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def main():
    print("=" * 65)
    print("TASK 3 - PREDICTIVE ANALYTICS USING HISTORICAL DATA")
    print("Sales Forecasting")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    print(f"\nDataset loaded: {len(df)} records")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    # Trend plot
    df["Rolling3"] = df["Sales"].rolling(3).mean()
    plt.figure(figsize=(11, 5))
    plt.plot(df["Date"], df["Sales"], marker="o", linewidth=2, label="Historical Sales")
    plt.plot(df["Date"], df["Rolling3"], linewidth=2, label="3-Month Moving Average")
    plt.title("Historical Monthly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "historical_sales_trend.png", dpi=160)
    plt.close()

    # Chronological split: first 80% train, final 20% test
    split = int(len(df) * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    features = ["TimeIndex", "MonthSin", "MonthCos", "Quarter"]
    X_train, y_train = train[features], train["Sales"]
    X_test, y_test = test[features], test["Sales"]

    # Baseline model
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)

    # Stronger model
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        min_samples_leaf=2
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    models = {
        "Linear Regression": linear_pred,
        "Random Forest": rf_pred
    }

    metrics = []
    for name, pred in models.items():
        mae, mse, rmse, r2 = evaluate(y_test, pred)
        metrics.append([name, mae, mse, rmse, r2])

    metrics_df = pd.DataFrame(
        metrics,
        columns=["Model", "MAE", "MSE", "RMSE", "R2"]
    )
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\nModel Evaluation:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # Pick model with lower RMSE
    best_name = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    best_pred = models[best_name]
    print(f"\nBest model based on RMSE: {best_name}")

    # Actual vs predicted
    plt.figure(figsize=(11, 5))
    plt.plot(test["Date"], y_test.values, marker="o", linewidth=2, label="Actual")
    plt.plot(test["Date"], best_pred, marker="s", linewidth=2, label=f"Predicted ({best_name})")
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Future forecast: next 6 months
    future_dates = pd.date_range(
        df["Date"].max() + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS"
    )
    future = pd.DataFrame({"Date": future_dates})
    future["Year"] = future["Date"].dt.year
    future["Month"] = future["Date"].dt.month
    future["Quarter"] = future["Date"].dt.quarter
    future["TimeIndex"] = np.arange(len(df), len(df) + len(future))
    future["MonthSin"] = np.sin(2 * np.pi * future["Month"] / 12)
    future["MonthCos"] = np.cos(2 * np.pi * future["Month"] / 12)

    # Retrain selected model on all historical data
    if best_name == "Random Forest":
        final_model = RandomForestRegressor(
            n_estimators=300, max_depth=8,
            random_state=42, min_samples_leaf=2
        )
    else:
        final_model = LinearRegression()

    final_model.fit(df[features], df["Sales"])
    future["PredictedSales"] = final_model.predict(future[features])
    future[["Date", "PredictedSales"]].to_csv(
        OUTPUT_DIR / "future_sales_forecast.csv", index=False
    )

    # Forecast plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Sales"], marker="o", linewidth=2, label="Historical Sales")
    plt.plot(
        future["Date"], future["PredictedSales"],
        marker="s", linestyle="--", linewidth=2,
        label="6-Month Forecast"
    )
    plt.axvline(df["Date"].max(), linestyle=":", linewidth=2, label="Forecast Start")
    plt.title("Sales Forecast - Next 6 Months")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sales_forecast.png", dpi=160)
    plt.close()

    print("\nNext 6 Months Forecast:")
    print(future[["Date", "PredictedSales"]].to_string(index=False))
    print("\nGenerated files:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {p.name}")

if __name__ == "__main__":
    main()
