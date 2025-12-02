# config.py
import os
from datetime import datetime, timedelta

# API Configuration
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
BINANCE_API_KEY = "your_binance_api_key"
BINANCE_SECRET_KEY = "your_binance_secret_key"

# Data Configuration
DEFAULT_CRYPTOS = ["bitcoin", "ethereum", "cardano", "solana", "polkadot"]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Model Configuration
TRAIN_TEST_SPLIT = 0.8
FORECAST_DAYS = 30

# Paths
DATA_PATH = "data/"
MODELS_PATH = "models/saved_models/"