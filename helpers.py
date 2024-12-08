import os
import pandas as pd
import numpy as np
import yfinance as yf

####
#### yfinance data fetching
####
 
DATA_FOLDER = "data/"
DATE_FORMAT = '%Y-%m-%d'

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

    return data

####
#### efficient frontier calcs
####

# Generate random portfolio weights and calculate risk & return
def random_portfolios(mean_returns, cov_matrix, num_portfolios=10000):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_stddev
        results[1, i] = portfolio_return
        results[2, i] = portfolio_return / portfolio_stddev  # Sharpe Ratio
        weights_record.append(weights)

    return results, weights_record

# Compute Efficient Frontier
def compute_efficient_frontier(mean_returns, cov_matrix, target_returns):
    num_assets = len(mean_returns)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(num_assets)

    a = np.dot(mean_returns.T, np.dot(inv_cov_matrix, mean_returns))
    b = np.dot(mean_returns.T, np.dot(inv_cov_matrix, ones))
    c = np.dot(ones.T, np.dot(inv_cov_matrix, ones))

    frontier_stddevs = []
    for target_return in target_returns:
        lambda_1 = (c * target_return - b) / (a * c - b ** 2)
        lambda_2 = (a - b * target_return) / (a * c - b ** 2)
        weights = lambda_1 * np.dot(inv_cov_matrix, mean_returns) + lambda_2 * np.dot(inv_cov_matrix, ones)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        frontier_stddevs.append(portfolio_stddev)

    return frontier_stddevs