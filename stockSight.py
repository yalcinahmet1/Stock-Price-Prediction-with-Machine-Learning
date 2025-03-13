import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# List of stock tickers
#tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'V']
tickers = ['AAPL']

frames = []
for ticker in tickers:
    try:
        # Fetch weekly data over a 10-year period
        data = yf.Ticker(ticker).history(period="max", interval="1wk")
        
        # Check if data is empty
        if data.empty:
            print(f"No data found for {ticker}. Skipping this ticker.")
            continue  # Skip to the next ticker if no data is found

        # Add ticker label and calculate features
        data['Ticker'] = ticker
        data['Next_Week_Change'] = data['Close'].shift(-1) - data['Close']
        data['Label'] = (data['Next_Week_Change'] > 0).astype(int)

        # Essential features for weekly data
        data['MA4'] = data['Close'].rolling(window=4).mean()  # 4-week moving average
        data['MA8'] = data['Close'].rolling(window=8).mean()  # 8-week moving average
        data['MA12'] = data['Close'].rolling(window=12).mean() # 12-week moving average
        data['Volatility'] = data['Close'].rolling(window=8).std()  # 8-week volatility
        
        # Bollinger Bands
        data['Bollinger_Upper'] = data['MA4'] + (2 * data['Volatility'])
        data['Bollinger_Lower'] = data['MA4'] - (2 * data['Volatility'])

        # RSI Calculation (14-week)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + gain / loss))

        # On-Balance Volume (OBV)
        obv = (data['Volume'] * ((data['Close'] - data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else -1))).cumsum()
        data['OBV'] = obv

        # MACD Calculation
        data['EMA4'] = data['Close'].ewm(span=4, adjust=False).mean()  # 4-week EMA
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()  # 8-week EMA
        data['MACD'] = data['EMA4'] - data['EMA12']  # MACD as the difference between EMA4 and EMA12

        # Lagged RSI and MACD
        data['Lagged_Close_1'] = data['Close'].shift(1)  # Lagged Close Price (1 week)
        data['Lagged_Volume_1'] = data['Volume'].shift(1)  # Lagged Volume (1 week)
        data['Lagged_OBV'] = data['OBV'].shift(1)  # Lagged OBV (1 week)
        data['Lagged_Close_1'] = data['Close'].shift(1)  # Lagged Close Price (1 week)
        data['Lagged_Volume_1'] = data['Volume'].shift(1)  # Lagged Volume (1 week)
        data['Lagged_OBV'] = data['OBV'].shift(1)  # Lagged OBV (1 week)
        data['Lagged_RSI'] = data['RSI'].shift(1)
        data['Lagged_MACD'] = data['MACD'].shift(1)
        
        # Stochastic Oscillator (%K and %D)
        data['Low14'] = data['Low'].rolling(14).min()
        data['High14'] = data['High'].rolling(14).max()
        data['%K'] = 100 * (data['Close'] - data['Low14']) / (data['High14'] - data['Low14'])
        data['%D'] = data['%K'].rolling(3).mean()  # 3-week moving average of %K

        # Momentum Indicator (10-week momentum)
        data['Momentum'] = data['Close'] - data['Close'].shift(10)

        # Price Rate of Change (ROC)
        data['ROC_4'] = data['Close'].pct_change(periods=4)  # 4-week rate of change
        data['ROC_12'] = data['Close'].pct_change(periods=12)  # 12-week rate of change

        # Chaikin Oscillator (3-week and 10-week)
        data['ADL'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
        data['Chaikin'] = data['ADL'].rolling(window=3).mean() - data['ADL'].rolling(window=10).mean()

        # Average Directional Index (ADX)
        up_move = data['High'] - data['High'].shift(1)
        down_move = data['Low'].shift(1) - data['Low']
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=data.index)
        tr = pd.concat([data['High'] - data['Low'], abs(data['High'] - data['Close'].shift(1)), abs(data['Low'] - data['Close'].shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        data['Plus_DI'] = 100 * (plus_dm.rolling(window=14).mean() / atr)
        data['Minus_DI'] = 100 * (minus_dm.rolling(window=14).mean() / atr)
        data['DX'] = 100 * abs(data['Plus_DI'] - data['Minus_DI']) / (data['Plus_DI'] + data['Minus_DI'])
        data['ADX'] = data['DX'].rolling(window=14).mean()

        # Williams %R (14-week)
        data['Williams_%R'] = (data['High14'] - data['Close']) / (data['High14'] - data['Low14']) * -100

        # Relative Volume (RVOL)
        data['RVOL'] = data['Volume'] / data['Volume'].rolling(window=4).mean()
        
        # Drop unnecessary columns
        data = data.drop(columns=['Dividends', 'Stock Splits', 'Adj Close', 'Price'], errors='ignore')

        # Reset index to make Date the first column
        data = data.reset_index()

        # Reorder columns to have Date, Ticker, then the rest
        columns_order = ['Date', 'Ticker'] + [col for col in data.columns if col not in ['Date', 'Ticker']]
        
        data = data.dropna()
        
        data = data[columns_order]
        # Initialize the scaler
        scaler = StandardScaler()

        # List of columns to scale (all columns except Date, Ticker, and Label)
        columns_to_scale = data.columns.difference(['Date', 'Ticker', 'Label'])
        
        # Apply Standard scaling to the selected columns
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
        
        frames.append(data)
        
    except Exception as e:
        print(f"An error occurred with ticker {ticker}: {e}")
        continue  # Skip to the next ticker in case of error

# Combine all data into one DataFrame without dropping NaNs
multi_stock_data = pd.concat(frames, ignore_index=True)

# Save the finalized dataset to CSV for easy inspection
multi_stock_data.to_csv('weekly_stock_data_finalized_cleaned.csv', index=False)

print("Final dataset saved to 'weekly_stock_data_finalized_cleaned.csv'")
