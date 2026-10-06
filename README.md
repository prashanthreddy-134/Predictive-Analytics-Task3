# Predictive Analytics Using Historical Data

## Task 3 — Sales Forecasting

A complete predictive analytics project that uses historical monthly sales data to identify trends, compare regression models, evaluate prediction accuracy, and forecast the next six months.

## Objective

Build a predictive model to forecast future trends using historical data while demonstrating data cleaning, preprocessing, model training, evaluation, and visualization.

## Key Features

- Historical sales data preprocessing
- Date/time feature engineering
- Historical trend visualization
- Chronological 80/20 train-test split
- Linear Regression baseline
- Random Forest Regression comparison
- MAE, MSE, RMSE, and R² evaluation
- Actual vs. predicted visualization
- Six-month future sales forecast
- CSV outputs for metrics and forecasts

## Technologies

- Python 3
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

## Project Structure

```text
Predictive_Analytics_Task3/
├── data/
│   └── historical_sales.csv
├── notebooks/
│   └── predictive_analytics.ipynb
├── outputs/
│   ├── historical_sales_trend.png
│   ├── actual_vs_predicted.png
│   ├── sales_forecast.png
│   ├── model_metrics.csv
│   └── future_sales_forecast.csv
├── src/
│   └── model.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run on Windows

### 1. Create a virtual environment

```powershell
python -m venv venv
```

### 2. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the project

```powershell
python main.py
```

The generated charts and CSV files are written to `outputs/`.

## Methodology

### Data preprocessing

The dataset is converted to chronological order, duplicate dates are removed, and missing values are checked. Time-based features are created from the date, including month, quarter, a sequential time index, and cyclic sine/cosine month features.

### Train-test strategy

Because the data is time ordered, the project does **not** randomly shuffle observations. The first 80% is used for training and the final 20% is held out for testing. This reduces the risk of using future observations to predict the past.

### Models

**1. Linear Regression**

Used as the baseline model because it is simple, interpretable, and provides a strong reference point for the forecasting task.

**2. Random Forest Regression**

Used as a nonlinear comparison model. The project compares both models using the same held-out test set.

### Evaluation metrics

- **MAE:** average absolute prediction error
- **MSE:** average squared prediction error
- **RMSE:** square root of MSE; lower is better
- **R²:** proportion of variance explained by the model; higher is generally better

The model with the lower test RMSE is selected for the final six-month forecast.

## Current Evaluation Result

With the included sample dataset, the test results are:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 250.31 | 310.38 | 0.81 |
| Random Forest | 708.42 | 844.20 | -0.40 |

Linear Regression is selected because it has the lower RMSE and substantially better test performance on this dataset.

## Forecast Output

The selected model is retrained using all available historical observations and used to estimate sales for the next six months. The forecast is saved in:

```text
outputs/future_sales_forecast.csv
```

## Dataset Note

The included dataset is a **synthetic educational dataset** containing 36 monthly observations from January 2023 through December 2025. It is intended to demonstrate the predictive analytics workflow.

For a production or real-business analysis, the dataset should be replaced with verified historical sales data while retaining the `Date` and `Sales` columns expected by the script.

## Limitations

- The included dataset is relatively small.
- The sample data is synthetic rather than real business data.
- Forecast accuracy can change substantially when real-world seasonality, promotions, holidays, pricing, inventory, and other external factors are introduced.
- A strong test score does not guarantee accurate future predictions outside the observed data distribution.

## Expected Learning Outcomes

This project demonstrates:

- Data cleaning and preprocessing
- Feature engineering
- Exploratory trend analysis
- Regression modeling
- Time-aware model evaluation
- Forecast generation
- Visualization of predictions
- Reproducible project organization with Git and GitHub

## Author

**Prashanth Reddy S**

Information Science and Engineering Student
