# src/data_collection.py
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

class CryptoDataCollector:
    def __init__(self):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
    def get_coingecko_data(self, crypto_id, vs_currency='usd', days='max'):
        """Fetch historical data from CoinGecko"""
        url = f"{self.coingecko_url}/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df = df[['price']]
            df.columns = [f'{crypto_id}_price']
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {crypto_id}: {e}")
            return None
    
    def get_yahoo_data(self, symbol, period="2y"):
        """Fetch data using yfinance"""
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            df = ticker.history(period=period)
            df = df[['Close']]
            df.columns = [f'{symbol}_price']
            return df
        except Exception as e:
            print(f"Error fetching Yahoo data for {symbol}: {e}")
            return None
    
    def get_multiple_cryptos(self, crypto_list, source='coingecko'):
        """Fetch data for multiple cryptocurrencies"""
        combined_df = pd.DataFrame()
        
        for crypto in crypto_list:
            print(f"Fetching data for {crypto}...")
            
            if source == 'coingecko':
                df = self.get_coingecko_data(crypto)
            else:
                df = self.get_yahoo_data(crypto.upper())
            
            if df is not None:
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = combined_df.join(df, how='outer')
            
            time.sleep(1)  # Rate limiting
        
        return combined_df