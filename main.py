import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from helpers import load_data

START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

WALLET1 = ["WOOD", "CUT", "IBB", "ERTH"]
WALLET2 = ["VAW", "IYM", "BAS.DE", "DOW", "2020.SR", "CTVA", "MOS", "CF", "UAN"]

def main():
    for ticker in WALLET1 + WALLET2:
        data = load_data(ticker, START_DATE, END_DATE)
        # print(data.head())

        print(f"Analyzing {ticker}")
        ticker_analysis(data, START_DATE, END_DATE)

    normalized_graphs(WALLET1)
    normalized_graphs(WALLET2)
    normalized_graphs(WALLET1 + WALLET2)


def ticker_analysis(data, start_date, end_date):
    data.index = pd.to_datetime(data.index)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

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

def return_rate(data):
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change()
    average_daily_return = df["Daily Return"].mean()
    
    annualized_return = (1 + average_daily_return) ** len(df) - 1

    return annualized_return * 100
        
def risk(data):
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change()
    risk = df["Daily Return"].std() * 100

    return risk

if __name__ == "__main__":
    main()