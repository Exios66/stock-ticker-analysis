import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt

# Function to load data and models would go here
# For simplicity, assuming models are saved and can be loaded

def main():
    st.title("Stock Market Forecasting App")
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    ticker = st.sidebar.selectbox("Select Stock Ticker", stock_tickers)
    
    # Display stock data
    st.write(f"### {ticker} Stock Price")
    data = yf.download(ticker, start='2015-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))
    st.line_chart(data['Close'])
    
    # Prediction section
    if st.button("Predict"):
        # Load the trained model
        model = all_models[ticker]['lstm_model']  # Example using LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data[ticker].dropna())
        
        # Prepare the last 60 days for prediction
        last_60_days = scaled_data[-60:]
        X_test = []
        X_test.append(last_60_days)
        X_test = np.array(X_test)
        
        # Make prediction
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(np.concatenate((np.zeros((1, scaled_data.shape[1]-1)), predicted_price), axis=1))[:, 3][0]
        
        st.write(f"### Predicted Close Price for Next Day: ${predicted_price:.2f}")

if __name__ == '__main__':
    main()
