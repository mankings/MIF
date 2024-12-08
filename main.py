import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from helpers import load_data, random_portfolios, compute_efficient_frontier

PLOT_FOLDER = "plots"

DATE_FORMAT = '%Y-%m-%d'
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
TRADING_DAYS = 252

WALLET1 = ["WOOD", "CUT", "IBB", "ERTH"]                                            # biological etf
WEIGHTS1 = [1/len(WALLET1) for _ in WALLET1]

WALLET2 = ["VAW", "IYM", "BAS.DE", "DOW", "2020.SR", "CTVA", "MOS", "CF", "UAN"]    # chemical etf
WEIGHTS2 = [1/len(WALLET2) for _ in WALLET2]

NUM_PORTFOLIOS = 5000

def main():
    print("\nWallet 1:\n")
    w1_data = [load_data(ticker, START_DATE, END_DATE) for ticker in WALLET1]
    w1 = list(zip(WALLET1, WEIGHTS1, w1_data))

    weighted_wallet_analysis(w1)
    normalized_graphs(w1, "wallet1_normalized.png")

    print("\nWallet 2:\n")
    w2_data = [load_data(ticker, START_DATE, END_DATE) for ticker in WALLET2]
    w2 = list(zip(WALLET2, WEIGHTS2, w2_data))

    weighted_wallet_analysis(w2)
    normalized_graphs(w2, "wallet2_normalized.png")
    
    print("------------------")

    print("Efficient Frontier1:\n")
    efficient_frontier_formula(w1)
    efficient_frontier_plot(w1, "wallet1_efficient_frontier.png")

    print("Efficient Frontier2:\n")

    efficient_frontier_formula(w2)
    efficient_frontier_plot(w2, "wallet2_efficient_frontier.png")

def efficient_frontier_plot(wallet_data, filename):
    r = np.array([return_rate(data) for _, _, data in wallet_data])
    cov_matrix = covariance_matrix(wallet_data)

    # Generate random portfolios    
    results, _ = random_portfolios(r, cov_matrix, num_portfolios=NUM_PORTFOLIOS)

    # Compute Efficient Frontier
    target_returns = np.linspace(min(results[1, :]), max(results[1, :]), 100)
    frontier_stddevs = compute_efficient_frontier(r, cov_matrix, target_returns)

    # Plot Efficient Frontier and random portfolios
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', alpha=0.5)
    plt.plot(frontier_stddevs, target_returns, 'r--', linewidth=2)
    plt.title('Efficient Frontier and Portfolio Scattering')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Return')
    # plt.colorbar(label='Sharpe Ratio')
    # plt.legend()
    plt.grid(True)

    # Save the plot
    file_path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(file_path, format="png")

    # Clear the plot for the next ticker
    plt.clf()
              
def efficient_frontier_formula(wallet_data):
    # Calculate the returns vector
    r = np.array([return_rate(data) for _, _, data in wallet_data])

    # Calculate the covariance matrix using annualized returns
    cov_matrix = covariance_matrix(wallet_data)

    # Unitary vector
    u = np.array([1 for _ in wallet_data])

    # Calculate the efficient frontier formula
    a1 = np.transpose(r) @ np.linalg.inv(cov_matrix) @ r
    a2 = np.transpose(r) @ np.linalg.inv(cov_matrix) @ u
    a3 = np.transpose(u) @ np.linalg.inv(cov_matrix) @ u

    d = a1 * a3 - a2 ** 2

    k1 = (a3 / d)
    k2 = (-2 * a2 / d)
    k3 = (a1 / d)

    print(f"Final formula: {k1} e^2 + {k2} e + {k3}")
    print(f"Return of the wallet with minimum risk: {a2/a3:.2f}")

def weighted_wallet_analysis(weighted_wallet_data):
    portfolio_returns = []
    portfolio_risks = []
    
    for _, weight, data in weighted_wallet_data:
        # Calculate individual return rate and risk
        rr = return_rate(data)
        r = risk(data)
        
        # Append weighted return and risk to lists
        portfolio_returns.append(rr * weight)
        portfolio_risks.append((r * weight) ** 2)  # Variance contribution for risk

    # Calculate portfolio metrics
    portfolio_return_rate = sum(portfolio_returns)
    portfolio_risk = (np.sqrt(sum(portfolio_risks)))  # Square root of sum of variances

    print(f"\tPortfolio Return Rate: {portfolio_return_rate:.2f}")
    print(f"\tPortfolio Risk: {portfolio_risk:.2f}\n")

def ticker_analysis(ticker_data, start_date, end_date):
    ticker_data.index = pd.to_datetime(ticker_data.index)

    start = datetime.strptime(start_date, DATE_FORMAT)
    end = datetime.strptime(end_date, DATE_FORMAT)

    if ticker_data is None:
        print("Error: No data available for the given ticker.")
        return

    current_year = start.year

    # Loop through each year in the given range
    while current_year <= end.year:
        year_start = datetime(current_year, 1, 1)
        year_end = datetime(current_year, 12, 31)

        # Filter data for the current year
        yearly_data = ticker_data[(ticker_data.index >= year_start) & (ticker_data.index <= year_end)]
        print(yearly_data.head(1))
        print(yearly_data.tail(1))
        print(len(yearly_data))

        # Calculate return rate and risk for the current year
        rr = return_rate(yearly_data)
        r = risk(yearly_data)

        print(f"  {current_year}, Return Rate: {rr:.2f} %, Risk: {r:.2f} %")
        current_year += 1

def normalized_graphs(wallet_data, filename):
        # Collect tickers from both wallets
    plt.figure(figsize=(12, 8))

    for ticker, weight, data in wallet_data:
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

    # Save the plot
    file_path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(file_path, format="png")
    
    # Clear the plot for the next ticker
    plt.clf()

def return_rate(data):
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change().dropna()

    # Calculate cumulative return over the entire period
    cumulative_return = (1 + df["Daily Return"]).prod() - 1

    # Annualize the cumulative return based on trading days (TRADING_DAYS days in a typical year)
    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(df.dropna())) - 1

    return annualized_return

def risk(data):
    df = data.copy()
    df["Daily Return"] = df["Close"].pct_change().dropna()

    # Calculate daily return standard deviation and annualize it
    daily_std_dev = df["Daily Return"].std()
    annualized_risk = daily_std_dev * np.sqrt(TRADING_DAYS)

    return annualized_risk

def covariance_matrix(wallet_data):
    # Calculate percentage change and align lengths
    returns = [
        data["Close"].pct_change().dropna().to_numpy() for _, _, data in wallet_data
    ]
    
    # Align lengths of returns
    min_length = min(len(ret) for ret in returns)
    aligned_returns = np.array([ret[-min_length:] for ret in returns])

    # Calculate the covariance matrix
    cov_matrix = np.cov(aligned_returns) * TRADING_DAYS

    return cov_matrix



if __name__ == "__main__":
    main()