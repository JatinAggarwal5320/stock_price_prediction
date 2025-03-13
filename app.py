import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Market Predictor")
st.markdown("### Predict future stock prices using deep learning (LSTM)")

stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2000-01-01"
end = "2025-03-01"

data = yf.download(stock, start, end)

st.markdown("## 📊 Stock Data Overview")
st.dataframe(data.tail())

data_train = pd.DataFrame(data["Close"][0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data["Close"][int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

ma_50 = data["Close"].rolling(50).mean()
ma_100 = data["Close"].rolling(100).mean()
ma_200 = data["Close"].rolling(200).mean()
ema_100 = data["Close"].ewm(span=100, adjust=False).mean()
ema_200 = data["Close"].ewm(span=200, adjust=False).mean()

st.markdown("## 📈 Moving Averages (MA)")
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(data["Close"], "g", label="Closing Price")
ax1.plot(ma_100, "r", label="100-Day MA")
ax1.plot(ma_200, "b", label="200-Day MA")
ax1.legend()
st.pyplot(fig1)

st.markdown("## 📉 Exponential Moving Averages (EMA)")
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(data["Close"], "g", label="Closing Price")
ax2.plot(ema_100, "r", label="100-Day EMA")
ax2.plot(ema_200, "b", label="200-Day EMA")
ax2.legend()
st.pyplot(fig2)

x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

model = load_model("/Users/atomic/Desktop/Python Projects/stock_pp/Stock Price Prediction Model.keras")
predicted_prices = model.predict(x)
predicted_prices = predicted_prices * (1 / scaler.scale_)
y = y * (1 / scaler.scale_)

mse = mean_squared_error(y, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, predicted_prices)
r2 = r2_score(y, predicted_prices) * 100

st.markdown("## 🔮 Predicted vs. Actual Prices")
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(y, "g", label="Actual Price")
ax3.plot(predicted_prices, "r", linestyle="dashed", label="Predicted Price")
ax3.legend()
st.pyplot(fig3)

st.markdown(f"### 🎯 Model Performance")
st.markdown(f"**MSE:** {mse:.2f} | **RMSE:** {rmse:.2f} | **MAE:** {mae:.2f} | **Accuracy (R² Score):** {r2:.2f}%")

latest_predicted_price = predicted_prices[-1][0]
st.markdown(f"### 💰 Predicted Closing Price: **${latest_predicted_price:.2f} USD**")

st.markdown("---")
st.markdown("**✨ Built by Jatin Aggarwal** 🚀")
