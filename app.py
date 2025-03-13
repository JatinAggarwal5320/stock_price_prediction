import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error ,r2_score


st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

model = load_model('/Users/atomic/Desktop/Python Projects/stock_pp/Stock Price Prediction Model.keras')

st.markdown(
    "<h1 style='text-align: center; color: #1B9CFC;'>Stock Market Predictor</h1>",
    unsafe_allow_html=True
)

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2005-01-01'
end = '2025-03-01'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = data.Close[:int(len(data) * 0.80)]
data_test = data.Close[int(len(data) * 0.80):]

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(np.array(data_test).reshape(-1, 1))

ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

ema_100_days = data.Close.ewm(span=100, adjust=False).mean()
ema_200_days = data.Close.ewm(span=200, adjust=False).mean()

fig1 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, 'g', label='Closing Price')
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.legend()
st.pyplot(fig1)

fig2 = plt.figure(figsize=(10, 5))
plt.plot(data.Close, 'g', label='Closing Price')
plt.plot(ema_100_days, 'r', label='EMA100')
plt.plot(ema_200_days, 'b', label='EMA200')
plt.legend()
st.pyplot(fig2)

x_test, y_test = [], []
for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
predicted_prices = model.predict(x_test)
predicted_prices = predicted_prices * (1 / scaler.scale_)

latest_predicted_price = predicted_prices[-1][0]

fig3 = plt.figure(figsize=(10, 5))
plt.plot(y_test * (1 / scaler.scale_), 'g', label='Original Price')
plt.plot(predicted_prices, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig3)

mse = mean_squared_error(y_test * (1 / scaler.scale_), predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test * (1 / scaler.scale_), predicted_prices)

st.markdown(f"""
    <h3 style='color: #1B9CFC;'>Model Performance</h3>
    <p><b>Mean Squared Error (MSE):</b> {mse:.2f}</p>
    <p><b>Root Mean Squared Error (RMSE):</b> {rmse:.2f}</p>
    <p><b>Mean Absolute Error (MAE):</b> {mae:.2f}</p>
""", unsafe_allow_html=True)

st.markdown(
    f"<h2 style='text-align: center; color: #2ed573;'>Predicted Price: ${latest_predicted_price:.2f} USD</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center; color: #FF4757;'>Build by Jatin Aggarwal</h4>",
    unsafe_allow_html=True
)

r2 = r2_score(y_test * (1 / scaler.scale_), predicted_prices)

st.markdown(f"""
    <h3 style='color: #1B9CFC;'>Model Performance</h3>
    <p><b>Mean Squared Error (MSE):</b> {mse:.2f}</p>
    <p><b>Root Mean Squared Error (RMSE):</b> {rmse:.2f}</p>
    <p><b>Mean Absolute Error (MAE):</b> {mae:.2f}</p>
    <p><b>R² Score (Accuracy):</b> {r2:.4f}</p>
""", unsafe_allow_html=True)
