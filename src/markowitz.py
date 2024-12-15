import os
import numpy as np
import matplotlib.pyplot as plt

from src.base_calcs import *
from scipy.optimize import minimize

PLOT_FOLDER = "plots"

import numpy as np
import matplotlib.pyplot as plt
import os

def efficient_frontier_sim(mean_returns, cov_matrix, tickers, filename):
    noPortfolios = 50000
    noAssets = len(mean_returns)

    # Initialize arrays for random portfolios
    weights = np.zeros((noPortfolios, noAssets))
    returns = np.zeros(noPortfolios)
    risks = np.zeros(noPortfolios)
    sharpes = np.zeros(noPortfolios)

    for k in range(noPortfolios):
        # Generate random weights
        w = np.random.random(noAssets)
        w = w / np.sum(w)  # Normalize weights to sum to 1
        weights[k, :] = w

        # Calculate expected return and risk
        returns[k] = np.dot(w.T, mean_returns)
        risks[k] = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        # Calculate Sharpe ratio (assuming risk-free rate is 0)
        sharpes[k] = returns[k] / risks[k]

    # Identify the portfolio with the maximum Sharpe ratio
    maxIndex = np.argmax(sharpes)
    maxSharpe = sharpes[maxIndex]
    optimalWeights = weights[maxIndex]

    ticker_weights = dict(zip(tickers, optimalWeights))

    print(f"Max Sharpe Ratio: {maxSharpe}")
    print("Optimal Weights by Ticker:")
    for ticker, weight in ticker_weights.items():
        print(f"\t{ticker}: {weight:.4f}")

    # Calculate single-asset portfolio metrics
    single_asset_returns = mean_returns
    single_asset_risks = np.sqrt(np.diag(cov_matrix))

    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    plt.scatter(risks, returns, c=sharpes, cmap='viridis', label="Random Portfolios")
    plt.colorbar(label='Sharpe Ratio')

    # Highlight the maximum Sharpe ratio portfolio
    plt.scatter(risks[maxIndex], returns[maxIndex], c='red', label="Max Sharpe Ratio")

    # Plot single-asset portfolios
    plt.scatter(single_asset_risks, single_asset_returns, c='blue', label="Single-Asset Portfolios", marker='X', s=100)

    # Add labels to single-asset portfolios
    for ticker, return_value, risk_value in zip(mean_returns.index, single_asset_returns, single_asset_risks):
        plt.text(risk_value, return_value, ticker, fontsize=10, ha='right')

    plt.xlabel('Expected Risk')
    plt.ylabel('Expected Log Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)

    # Save the plot
    PLOT_FOLDER = "plots"  # You can change this to your desired directory
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)

    file_path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(file_path, format="png")

    return optimalWeights, maxSharpe

def optimize_max_sharpe(mean_returns, cov_matrix, tickers):
    noAssets = len(mean_returns)

    def sharpe_ratio_neg(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_risk  # Negative Sharpe Ratio for minimization

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(noAssets))

    # Initial guess (equal weights)
    initial_weights = np.ones(noAssets) / noAssets

    result = minimize(sharpe_ratio_neg, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        optimal_sharpe = -result.fun  # Convert back to positive Sharpe Ratio
        print(f"Max Sharpe Ratio: {optimal_sharpe}")

        expected_return = np.dot(optimal_weights, mean_returns)
        print(f"Expected return of these weights: {expected_return}")
        expected_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        print(f"Expected risk of these weights: {expected_risk}")
        
        # Pair tickers with weights
        ticker_weights = dict(zip(tickers, optimal_weights))

        print("Optimal Weights by Ticker:")
        for ticker, weight in ticker_weights.items():
            print(f"\t{ticker}: {weight:.4f}")
        return optimal_weights, optimal_sharpe
    else:
        raise ValueError("Optimization did not converge")


def optimize_min_risk(mean_returns, cov_matrix, tickers):
    noAssets = len(mean_returns)

    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio risk

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(noAssets))

    # Initial guess (equal weights)
    initial_weights = np.ones(noAssets) / noAssets

    result = minimize(portfolio_risk, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x

        min_risk = result.fun  # Minimum risk
        print(f"Minimum Risk: {min_risk}")

        expected_return = np.dot(optimal_weights, mean_returns)
        print(f"Expected return of these weights: {expected_return}")

        sharpeRatio = expected_return / min_risk
        print(f"Sharpe Ratio of these weights: {sharpeRatio}")

        # Pair tickers with weights
        ticker_weights = dict(zip(tickers, optimal_weights))

        print("Optimal Weights by Ticker:")
        for ticker, weight in ticker_weights.items():
            print(f"\t{ticker}: {weight:.4f}")

        return ticker_weights, min_risk
    else:
        raise ValueError("Optimization did not converge")