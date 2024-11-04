import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from helpers import load_data


DATE_FORMAT = '%Y-%m-%d'
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

WALLET1 = ["WOOD", "CUT", "IBB", "ERTH"]
WALLET2 = ["VAW", "IYM", "BAS.DE", "DOW", "2020.SR", "CTVA", "MOS", "CF", "UAN"]

def main():
    w1 = [(ticker, 1/len(WALLET1)) for ticker in WALLET1]
    w2 = [(ticker, 1/len(WALLET2)) for ticker in WALLET2]

    weighted_wallet_analysis(w1, START_DATE, END_DATE)
    weighted_wallet_analysis(w2, START_DATE, END_DATE)

    normalized_graphs(WALLET1)
    normalized_graphs(WALLET2)

def wallet_analysis(wallet, start_date, end_date):
    for ticker in wallet:
        data = load_data(ticker, start_date, end_date)
        print(f"Analyzing {ticker}")
        ticker_analysis(data, start_date, end_date)

def weighted_wallet_analysis(weighted_wallet, start_date, end_date):
    portfolio_returns = []
    portfolio_risks = []
    
    for ticker, weight in weighted_wallet:
        data = load_data(ticker, start_date, end_date)
        if data is None:
            print(f"Error: No data for {ticker}. Skipping.")
            continue

        # Calculate individual return rate and risk
        rr = return_rate(data)
        r = risk(data)
        
        # Append weighted return and risk to lists
        portfolio_returns.append(rr * weight)
        portfolio_risks.append((r * weight) ** 2)  # Variance contribution for risk

    # Calculate portfolio metrics
    portfolio_return_rate = sum(portfolio_returns)
    portfolio_risk = (sum(portfolio_risks) ** 0.5)  # Square root of sum of variances

    print(f"Portfolio Return Rate: {portfolio_return_rate:.2f} %")
    print(f"Portfolio Risk: {portfolio_risk:.2f} %")

def ticker_analysis(data, start_date, end_date):
    data.index = pd.to_datetime(data.index)

    start = datetime.strptime(start_date, DATE_FORMAT)
    end = datetime.strptime(end_date, DATE_FORMAT)

    if data is None:
        print("Error: No data available for the given ticker.")
        return

    current_year = start.year

    # Loop through each year in the given range
    while current_year <= end.year:
        year_start = datetime(current_year, 1, 1)
        year_end = datetime(current_year, 12, 31)

        # Filter data for the current year
        yearly_data = data[(data.index >= year_start) & (data.index <= year_end)]
        print(yearly_data.head(1))
        print(yearly_data.tail(1))
        print(len(yearly_data))

        # Calculate return rate and risk for the current year
        rr = return_rate(yearly_data)
        r = risk(yearly_data)

        print(f"  {current_year}, Return Rate: {rr:.2f} %, Risk: {r:.2f} %")
        current_year += 1

def normalized_graphs(tickers):
        # Collect tickers from both wallets
    plt.figure(figsize=(12, 8))

    for ticker in tickers:
        # Load data
        data = load_data(ticker, START_DATE, END_DATE)
        if data is None or 'Close' not in data.columns:
            print(f"No data for {ticker}")
            continue

        # Convert index to datetime if necessary
        data.index = pd.to_datetime(data.index)

        # Normalize the 'Close' prices
        normalized_prices = data['Close'] / data['Close'].iloc[0]

        # Plot normalized prices
        plt.plot(data.index, normalized_prices, label=ticker)

    # Graph formatting
    plt.xlabel('Year')
    plt.ylabel('Normalized Price')
    plt.title('Normalized Price Evolution Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# def return_rate(data):
#     df = data.copy()
#     df["Daily Return"] = df["Close"].pct_change()
#     average_daily_return = df["Daily Return"].mean()
    
#     annualized_return = (1 + average_daily_return) ** len(df) - 1

#     return annualized_return * 100
        
# def risk(data):
#     df = data.copy()
#     df["Daily Return"] = df["Close"].pct_change()
#     risk = df["Daily Return"].std() * 100

#     return risk

def return_rate(data):
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change()

    # Calculate cumulative return over the entire period
    cumulative_return = (1 + df["Daily Return"]).prod() - 1

    # Annualize the cumulative return based on trading days (252 days in a typical year)
    annualized_return = (1 + cumulative_return) ** (252 / len(df.dropna())) - 1

    return annualized_return * 100

def risk(data):
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change()

    # Calculate daily return standard deviation and annualize it
    daily_std_dev = df["Daily Return"].std()
    annualized_risk = daily_std_dev * np.sqrt(252)

    return annualized_risk * 100

if __name__ == "__main__":
    main()