import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.title("📊 Walmart Sales Forecasting App")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    df = df.groupby(df.index).sum()
    df["Weekly_Sales"] = df["Weekly_Sales"].ffill()

    st.sidebar.success("✅ File Uploaded Successfully!")

    # 📊 Plot sales trend
    st.subheader("📈 Sales Trend Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Weekly_Sales"], color="blue", label="Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.set_title("Sales Trend Over Time")
    ax.legend()
    st.pyplot(fig)

    # 📌 ARIMA Hyperparameter Tuning (Grid Search)
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    best_aic = float("inf")
    best_order = None

    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(df["Weekly_Sales"], order=(p, d, q))
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
        except:
            continue

    st.subheader("🔍 Best ARIMA Order")
    st.write(f"✅ Best ARIMA Order: {best_order} with AIC: {best_aic}")

    # 🚀 Fit ARIMA Model
    model = ARIMA(df["Weekly_Sales"], order=best_order)
    arima_result = model.fit()

    # 📊 Evaluate ARIMA Performance
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    train_model = ARIMA(train["Weekly_Sales"], order=best_order).fit()
    test_forecast = train_model.forecast(steps=len(test))
    rmse_arima = np.sqrt(mean_squared_error(test["Weekly_Sales"], test_forecast))
    st.write(f"📊 **ARIMA RMSE:** {rmse_arima}")

    # 🔍 Prepare Data for XGBoost
    lags = 10
    for i in range(1, lags + 1):
        df[f"Lag_{i}"] = df["Weekly_Sales"].shift(i)
    df.dropna(inplace=True)

    X = df.drop(columns=["Weekly_Sales"], errors='ignore')
    y = df["Weekly_Sales"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 📌 Train XGBoost Model
    xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train_scaled, y_train)

    # 📊 Evaluate XGBoost Model
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    st.write(f"📊 **XGBoost RMSE:** {rmse_xgb}")

    # 📊 Forecast Future Sales
    forecast_steps = 10
    arima_forecast = arima_result.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq="W-FRI")[1:]

    st.subheader("📈 Forecasted Sales")
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted_Sales": arima_forecast})
    st.write(forecast_df)

    # 📊 Plot Forecast
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Weekly_Sales"], color="blue", label="Actual Sales")
    ax.plot(forecast_dates, arima_forecast, "r--", label="Forecasted Sales")
    ax.axvline(df.index[-1], color="black", linestyle="dotted", label="Forecast Start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Forecasted Sales")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("⚠️ Please upload a dataset to proceed.")