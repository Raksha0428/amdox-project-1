import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf  # ← Added import

# Add the function here
def get_real_crypto_data(symbol, period="6mo"):
    """Fetch real cryptocurrency data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        data = ticker.history(period=period)
        if data.empty:
             st.warning(f"No data found for {symbol}")
             return None
        return data[['Close']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None


# Page configuration
st.set_page_config(
    page_title="Cryptocurrency Time Series Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Cryptocurrency Time Series Analysis")
st.markdown("""
This application provides analysis and forecasting of cryptocurrency prices 
using time series models.
""")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Cryptocurrency selection
crypto_options = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH", 
    "Cardano": "ADA",
    "Solana": "SOL",
    "Dogecoin": "DOGE",
    "Litecoin":"LTC"
}

selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency:",
    list(crypto_options.keys())
)

# Forecast days
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model:",
    ["ARIMA", "LSTM", "Compare Both"]
)

# Generate realistic sample data
def generate_sample_data(days=365):
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=days)
    
    # Simulate realistic crypto price patterns
    prices = [40000]  # Starting price for Bitcoin
    volatility = 0.04  # 4% daily volatility
    
    for i in range(1, days):
        # Random walk with some trend and volatility
        change = np.random.normal(0.001, volatility)
        new_price = prices[-1] * (1 + change)
        
        # Ensure prices don't go negative
        if new_price <= 0:
            new_price = prices[-1] * 0.99
            
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    df.set_index('Date', inplace=True)
    return df

# Simple forecasting functions for demo
def simple_arima_forecast(data, steps=30):
    """Simple ARIMA-like forecast for demo"""
    last_price = data['Price'].iloc[-1]
    trend = (data['Price'].iloc[-1] - data['Price'].iloc[-30]) / 30
    forecast = []
    
    for i in range(steps):
        # Add trend with some noise
        next_price = last_price + trend + np.random.normal(0, last_price * 0.01)
        forecast.append(next_price)
        last_price = next_price
    
    return np.array(forecast)

def simple_lstm_forecast(data, steps=30):
    """Simple LSTM-like forecast for demo"""
    last_price = data['Price'].iloc[-1]
    # Use recent momentum
    recent_trend = (data['Price'].iloc[-1] - data['Price'].iloc[-7]) / 7
    forecast = []
    
    for i in range(steps):
        # More complex pattern for LSTM simulation
        noise = np.random.normal(0, last_price * 0.008)
        next_price = last_price + recent_trend * 0.7 + noise
        forecast.append(next_price)
        last_price = next_price
        recent_trend = recent_trend * 0.95  # Decaying trend
    
    return np.array(forecast)

# Main content
if st.sidebar.button("Generate Analysis"):
    with st.spinner("Fetching cryptocurrency data..."):
        
        try:
            # Fetch real cryptocurrency data
            symbol = crypto_options[selected_crypto]
            real_data = get_real_crypto_data(symbol, period="6mo")
            
            if real_data is not None and not real_data.empty:
                # Use real data
                data = real_data.rename(columns={'Close': 'Price'})
                st.success(f"✅ Successfully fetched real {selected_crypto} data!")
            else:
                # Fallback to sample data
                data = generate_sample_data()
                st.warning("Using sample data (real data unavailable)")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            # Fallback to sample data
            data = generate_sample_data()
        
        # Continue with the rest of your analysis
        st.subheader(f"{selected_crypto} Price Data")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(data.tail(10), use_container_width=True)
        
        with col2:
            current_price = data['Price'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
            
            # Calculate 24h change
            if len(data) > 1:
                change_24h = ((data['Price'].iloc[-1] - data['Price'].iloc[-2]) / data['Price'].iloc[-2]) * 100
                st.metric("24h Change", f"{change_24h:+.2f}%")
            else:
                st.metric("24h Change", f"+{np.random.uniform(0.1, 5.0):.2f}%")
            
            st.metric("Market Cap", f"${np.random.randint(500, 1000)}B")
        
        # Price chart
    st.subheader("Price History")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
            x=data.index,
            y=data['Price'],
            mode='lines',
            name=f'{selected_crypto} Price',
            line=dict(color='#00D4AA', width=2)
        ))
    fig_price.update_layout(
            title=f"{selected_crypto} Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=400
        )
    st.plotly_chart(fig_price, use_container_width=True)
        
        # Forecasting
    st.subheader("Price Forecasting")
        
    forecasts = {}
        
    if model_choice in ["ARIMA", "Compare Both"]:
            # ARIMA Forecasting
            arima_forecast = simple_arima_forecast(data, steps=forecast_days)
            forecasts['ARIMA'] = arima_forecast
        
    if model_choice in ["LSTM", "Compare Both"]:
            # LSTM Forecasting
            lstm_forecast = simple_lstm_forecast(data, steps=forecast_days)
            forecasts['LSTM'] = lstm_forecast
        
        # Plot forecasts
    fig_forecast = go.Figure()
        
        # Historical data
    fig_forecast.add_trace(go.Scatter(
            x=data.index,
            y=data['Price'],
            name='Historical Price',
            line=dict(color='#00D4AA', width=2)
        ))
        
        # Forecasts
    colors = {'ARIMA': '#FF6B6B', 'LSTM': '#4ECDC4'}
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
        
    for model_name, forecast in forecasts.items():
            fig_forecast.add_trace(go.Scatter(
                x=future_dates,
                y=forecast,
                name=f'{model_name} Forecast',
                line=dict(color=colors.get(model_name, '#FFE66D'), dash='dash', width=2)
            ))
        
    fig_forecast.update_layout(
            title=f"{selected_crypto} Price Forecast ({model_choice})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500
        )
        
    st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast statistics
    st.subheader("Forecast Summary")
    forecast_df = pd.DataFrame({'Date': future_dates})
        
    for model_name, forecast in forecasts.items():
            forecast_df[model_name] = forecast
        
    st.dataframe(forecast_df.style.format({
            'ARIMA': '${:.2f}',
            'LSTM': '${:.2f}'
        }), use_container_width=True)
        
    st.success("✅ Analysis completed successfully!")

# Instructions
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
1. Select a cryptocurrency
2. Choose forecast period
3. Select forecasting model
4. Click 'Generate Analysis' button
""")

# Footer
st.markdown("---")
st.markdown("*Note: This is a demo application with simulated data*")