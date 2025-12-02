# src/models/arima_model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self):
        self.model = None
        self.model_fit = None
        
    def check_stationarity(self, series):
        """Check stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    
    def auto_arima(self, series, seasonal=False, m=7):
        """Automatically find best ARIMA parameters"""
        model = pm.auto_arima(
            series,
            seasonal=seasonal,
            m=m,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,  # Let model determine
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        return model
    
    def fit(self, series, order=None):
        """Fit ARIMA model"""
        if order is None:
            print("Finding optimal ARIMA parameters...")
            auto_model = self.auto_arima(series)
            order = auto_model.order
            print(f"Optimal order: {order}")
        
        self.model = ARIMA(series, order=order)
        self.model_fit = self.model.fit()
        return self.model_fit
    
    def forecast(self, steps=30):
        """Generate forecast"""
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast
    
    def evaluate(self, test_series):
        """Evaluate model performance"""
        if self.model_fit is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Generate predictions for test period
        predictions = self.model_fit.forecast(steps=len(test_series))
        
        mse = mean_squared_error(test_series, predictions)
        mae = mean_absolute_error(test_series, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions
        }