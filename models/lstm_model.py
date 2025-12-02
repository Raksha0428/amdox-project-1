# src/models/lstm_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data):
        """Prepare data for LSTM"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, train_data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        X_train, y_train = self.prepare_data(train_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        self.model = self.build_model((X_train.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=False
        )
        
        return history
    
    def predict(self, data):
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare the data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        X = []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
        
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def forecast_future(self, last_sequence, steps=30):
        """Forecast future values"""
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        current_sequence = last_sequence.copy()
        predictions = []
        
        for _ in range(steps):
            # Prepare input
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(X_input, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()