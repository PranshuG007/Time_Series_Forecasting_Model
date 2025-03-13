# 📊 Walmart Sales Forecasting App

## 📌 Overview

This Streamlit app predicts Walmart's weekly sales using **ARIMA** and **XGBoost** models. It allows users to:
✅ Upload a CSV dataset 📂\
✅ Visualize historical sales trends 📈\
✅ Find the best ARIMA parameters automatically 🔍\
✅ Compare **ARIMA vs XGBoost** performance based on RMSE 📊\
✅ Forecast future sales 🔮

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/PranshuG007/Walmart_Sales_Forecasting.git
cd Walmart_Sales_Forecasting
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```sh
streamlit run load.py
```

Replace `your_script.py` with the actual filename.

---

## 📂 Expected CSV Format

Upload a CSV file with these columns:

| Date       | Store | Weekly\_Sales | Holiday\_Flag | Temperature | Fuel\_Price | CPI | Unemployment |
| ---------- | ----- | ------------- | ------------- | ----------- | ----------- | --- | ------------ |
| 2010-02-05 | 1     | 1643690.9     | 0             | 42.31       | 2.572       | 211 | 8.106        |

The **Date** column must be in `YYYY-MM-DD` format.

---

## 🏆 Model Performance

- **ARIMA Model:** Finds optimal `(p, d, q)` using **AIC minimization**.
- **XGBoost Model:** Uses **previous 10 weeks’ sales** as features.
- **RMSE Comparison:** Displays **error rates** to compare models.

---

## 🎯 Features

- 📌 **Automated ARIMA tuning** (Grid Search for best `(p, d, q)`).
- 🔥 **XGBoost for tree-based forecasting**.
- 📊 **Dynamic RMSE evaluation** for model performance.
- 🖼️ **Interactive visualizations** of sales & forecasts.

---

## 🏗️ Contributing

Contributions are welcome! Feel free to fork, improve, and submit a PR.

---

## 📜 License

This project is open-source under the **MIT License**.

🚀 **Happy Forecasting!**

