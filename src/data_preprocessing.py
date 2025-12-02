# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
    
    def handle_missing_values(self, df, method='ffill'):
        """Handle missing values in time series data"""
        df_clean = df.copy()
        
        # Forward fill, then backward fill any remaining NaNs
        df_clean = df_clean.ffill().bfill()
        
        return df_clean
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for time series analysis"""
        df_indicators = df.copy()
        
        for column in df.columns:
            # Rolling statistics
            df_indicators[f'{column}_rolling_mean_7'] = df[column].rolling(window=7).mean()
            df_indicators[f'{column}_rolling_std_7'] = df[column].rolling(window=7).std()
            
            # Price changes
            df_indicators[f'{column}_daily_return'] = df[column].pct_change()
            df_indicators[f'{column}_volatility'] = df[column].pct_change().rolling(window=7).std()
            
            # Moving averages
            df_indicators[f'{column}_sma_20'] = df[column].rolling(window=20).mean()
            df_indicators[f'{column}_sma_50'] = df[column].rolling(window=50).mean()
        
        return df_indicators
    
    def scale_data(self, df, columns=None):
        """Scale data using MinMaxScaler"""
        if columns is None:
            columns = df.columns
        
        scaled_data = df.copy()
        
        for column in columns:
            if df[column].isna().any():
                continue
                
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(df[[column]])
            scaled_data[column] = scaled_values.flatten()
            self.scalers[column] = scaler
        
        return scaled_data
    
    def prepare_lstm_data(self, df, target_column, sequence_length=60):
        """Prepare data for LSTM model"""
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in DataFrame")
        
        # Remove rows with NaN in target column
        df_clean = df.dropna(subset=[target_column])
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(df_clean)):
            X.append(df_clean[target_column].iloc[i-sequence_length:i].values)
            y.append(df_clean[target_column].iloc[i])
        
        return np.array(X), np.array(y)