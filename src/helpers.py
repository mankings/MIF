import os
import pandas as pd
import yfinance as yf

####
#### yfinance data fetching
####
 
DATA_FOLDER = "data/"
DATE_FORMAT = '%Y-%m-%d'

def load_wallet_data(wallet, start, end, filename):
    data = None
    file = f"{DATA_FOLDER}{filename}.csv"

    if os.path.exists(file):
        # print(f"Loading data from {file}...")
        data = pd.read_csv(file, parse_dates=True, index_col=0)
    
    else:
        # print(f"Fetching data for wallet using yfinance...")
        data = fetch_wallet_data(wallet, start, end)
        data.to_csv(file)

    return data

def fetch_wallet_data(wallet, start_date, end_date):
    data = yf.download(wallet, start=start_date, end=end_date)["Close"]
    data.index = data.index.strftime(DATE_FORMAT)

    return data

def load_data(ticker, start, end):
    data = None
    file = f"{DATA_FOLDER}{ticker}.csv"

    if os.path.exists(file):
        # print(f"Loading data from {file}...")
        data = pd.read_csv(file, parse_dates=True, index_col=0)

    else: 
        # print(f"Fetching data for {ticker} using yfinance...")
        data = fetch_data(ticker, start, end)
        data.to_csv(file)

    return data

def fetch_data(ticker, start_date, end_date, interval="1d"):
    ticker = yf.Ticker(ticker)
    data = ticker.history(start=start_date, end=end_date, interval=interval)

    if data.empty:
        print(f"Error: No data available for {ticker}.")
        return None

    # clean the data a bit
    data.index = data.index.strftime(DATE_FORMAT)

    return data["Close"]

