import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Sales Predictive Analytics", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "historical_sales.csv"

st.title("📈 Predictive Analytics Using Historical Data")
st.subheader("Sales Forecasting Dashboard")
st.write("Analyze historical sales, compare regression models, evaluate performance, and forecast future sales.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

def create_features(data):
    data = data.copy()
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter
    data["TimeIndex"] = np.arange(len(data))
    data["MonthSin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["MonthCos"] = np.cos(2 * np.pi * data["Month"] / 12)
    return data

df = create_features(load_data())

st.sidebar.header("Dashboard Controls")
forecast_months = st.sidebar.slider("Forecast Months", 3, 12, 6)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical Records", len(df))
c2.metric("Average Sales", f"{df['Sales'].mean():,.0f}")
c3.metric("Highest Sales", f"{df['Sales'].max():,.0f}")
c4.metric("Lowest Sales", f"{df['Sales'].min():,.0f}")

st.markdown("---")
st.header("📊 Historical Sales Trend")

df["MovingAverage"] = df["Sales"].rolling(3).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df["Sales"], marker="o", linewidth=2, label="Historical Sales")
ax.plot(df["Date"], df["MovingAverage"], linestyle="--", linewidth=2, label="3-Month Moving Average")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Historical Monthly Sales")
ax.grid(alpha=0.3)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

with st.expander("View Historical Dataset"):
    st.dataframe(df[["Date", "Sales"]], use_container_width=True)

features = ["TimeIndex", "MonthSin", "MonthCos", "Quarter"]

split = int(len(df) * 0.80)
train = df.iloc[:split]
test = df.iloc[split:]

linear = LinearRegression()
linear.fit(train[features], train["Sales"])
linear_prediction = linear.predict(test[features])

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    min_samples_leaf=2
)
rf.fit(train[features], train["Sales"])
rf_prediction = rf.predict(test[features])

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R²": r2_score(actual, predicted)
    }

linear_metrics = calculate_metrics(test["Sales"], linear_prediction)
rf_metrics = calculate_metrics(test["Sales"], rf_prediction)

metrics = pd.DataFrame([
    {"Model": "Linear Regression", **linear_metrics},
    {"Model": "Random Forest", **rf_metrics}
])

st.markdown("---")
st.header("🤖 Model Evaluation")

st.dataframe(
    metrics.style.format({
        "MAE": "{:,.2f}",
        "MSE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R²": "{:.2f}"
    }),
    use_container_width=True
)

if linear_metrics["RMSE"] <= rf_metrics["RMSE"]:
    best_name = "Linear Regression"
    best_prediction = linear_prediction
    best_rmse = linear_metrics["RMSE"]
    best_r2 = linear_metrics["R²"]
    final_model = LinearRegression()
else:
    best_name = "Random Forest"
    best_prediction = rf_prediction
    best_rmse = rf_metrics["RMSE"]
    best_r2 = rf_metrics["R²"]
    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        min_samples_leaf=2
    )

st.success(
    f"Best Model: {best_name} | RMSE: {best_rmse:,.2f} | R²: {best_r2:.2f}"
)

st.header("🎯 Actual vs Predicted Sales")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(
    test["Date"],
    test["Sales"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)
ax2.plot(
    test["Date"],
    best_prediction,
    marker="s",
    linestyle="--",
    linewidth=2,
    label=f"Predicted ({best_name})"
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")
ax2.set_title("Actual vs Predicted Sales")
ax2.grid(alpha=0.3)
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")
st.header("🔮 Future Sales Forecast")

future_dates = pd.date_range(
    df["Date"].max() + pd.offsets.MonthBegin(1),
    periods=forecast_months,
    freq="MS"
)

future = pd.DataFrame({"Date": future_dates})
future["Month"] = future["Date"].dt.month
future["Quarter"] = future["Date"].dt.quarter
future["TimeIndex"] = np.arange(len(df), len(df) + forecast_months)
future["MonthSin"] = np.sin(2 * np.pi * future["Month"] / 12)
future["MonthCos"] = np.cos(2 * np.pi * future["Month"] / 12)

final_model.fit(df[features], df["Sales"])
future["PredictedSales"] = final_model.predict(future[features])

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(
    df["Date"],
    df["Sales"],
    marker="o",
    linewidth=2,
    label="Historical Sales"
)
ax3.plot(
    future["Date"],
    future["PredictedSales"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Forecast"
)
ax3.axvline(
    df["Date"].max(),
    linestyle=":",
    linewidth=2,
    label="Forecast Start"
)
ax3.set_xlabel("Date")
ax3.set_ylabel("Sales")
ax3.set_title(f"Sales Forecast - Next {forecast_months} Months")
ax3.grid(alpha=0.3)
ax3.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

forecast_display = future[["Date", "PredictedSales"]].copy()
forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m-%d")
forecast_display["PredictedSales"] = forecast_display["PredictedSales"].round(2)

st.subheader("Forecast Values")
st.dataframe(forecast_display, use_container_width=True)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Forecast",
    f"{future['PredictedSales'].mean():,.0f}"
)

c2.metric(
    "Highest Forecast",
    f"{future['PredictedSales'].max():,.0f}"
)

c3.metric(
    "Lowest Forecast",
    f"{future['PredictedSales'].min():,.0f}"
)

st.download_button(
    "⬇️ Download Forecast CSV",
    future[["Date", "PredictedSales"]].to_csv(index=False),
    "future_sales_forecast.csv",
    "text/csv"
)

st.markdown("---")

st.info(
    "Dataset note: the included historical sales data is synthetic educational data. "
    "It should not be represented as real company sales data."
)

st.caption(
    "Predictive Analytics Task 3 | Sales Forecasting | Python + Scikit-learn + Streamlit"
)
