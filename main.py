from src.helpers import *
from src.wallets import *
from src.markowitz import *

START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
TRADING_DAYS = 252

WALLET1 = ["WOOD", "CUT", "IBB", "ERTH"]                       # biological etf
WALLET2 = ["VAW", "IYM", "DOW", "CTVA", "MOS", "CF", "UAN"]    # chemical etf
FULL_WALLET = WALLET1 + WALLET2

def main():
    ## data fetching
    print("\nLoading Wallet 1")
    close1 = load_wallet_data(WALLET1, START_DATE, END_DATE, "wallet1")
    normalized_graphs(close1, "wallet1_normalized.png")
    print("\nSimulating wallet1 efficient frontier...")
    daily_returns1 = close1.pct_change(fill_method=None).dropna()
    mean_returns1 = daily_returns1.mean() * TRADING_DAYS
    cov_matrix1 = daily_returns1.cov() * TRADING_DAYS
    op1 = efficient_frontier_sim(mean_returns1, cov_matrix1, daily_returns1.columns, "wallet1_efficient_frontier.png")
    print("\nOptimizing wallet1 weights...")
    optimized_weights1a, optimal_sharpe1 = optimize_max_sharpe(mean_returns1, cov_matrix1, close1.columns)
    optimized_weights1b, min_risk1 = optimize_min_risk(mean_returns1, cov_matrix1, close1.columns)

    print("\nLoading Wallet 2")
    close2 = load_wallet_data(WALLET2, START_DATE, END_DATE, "wallet2")
    normalized_graphs(close2, "wallet2_normalized.png")
    print("\nSimulating wallet2 efficient frontier...")
    daily_returns2 = close2.pct_change(fill_method=None).dropna()
    mean_returns2 = daily_returns2.mean() * TRADING_DAYS
    cov_matrix2 = daily_returns2.cov() * TRADING_DAYS
    op2 = efficient_frontier_sim(mean_returns2, cov_matrix2, daily_returns2.columns, "wallet2_efficient_frontier.png")
    print("\nOptimizing wallet2 weights...")
    optimized_weights2a, optimal_sharpe2 = optimize_max_sharpe(mean_returns2, cov_matrix2, close2.columns)
    optimized_weights2b, min_risk2 = optimize_min_risk(mean_returns2, cov_matrix2, close2.columns)

    print("\nLoading Full Wallet")
    fullClose = close1.join(close2, how="outer")
    print("\nSimulating joint wallet efficient frontier...")
    full_daily_returns = fullClose.pct_change(fill_method=None).dropna()
    full_mean_returns = full_daily_returns.mean() * TRADING_DAYS
    full_cov_matrix = full_daily_returns.cov() * TRADING_DAYS
    opf = efficient_frontier_sim(full_mean_returns, full_cov_matrix, full_daily_returns.columns, "full_wallet_efficient_frontier.png")
    print("\nOptimizing joint wallet weights...")
    optimized_weightsa, optimal_sharpe = optimize_max_sharpe(full_mean_returns, full_cov_matrix, fullClose.columns)
    optimized_weightsb, min_risk = optimize_min_risk(full_mean_returns, full_cov_matrix, fullClose.columns)

if __name__ == "__main__":
    main()