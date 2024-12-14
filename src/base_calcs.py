import numpy as np

TRADING_DAYS = 252

def get_stats(returns):
    return return_rate_log(returns), risk(returns)

def return_rate(returns):
    df = returns.copy()
    df = df.pct_change().dropna()

    # Calculate cumulative return over the entire period
    cumulative_return = (1 + df).prod() - 1

    # Annualize the cumulative return based on trading days (TRADING_DAYS days in a typical year)
    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / len(df.dropna())) - 1

    return annualized_return

def return_rate_log(returns):
    df = returns.copy()

    # Calculate daily log returns
    df = np.log(df / df.shift(1)).dropna()

    # Calculate the average daily log return
    avg_daily_log_return = df.mean()

    # Annualize the log return
    annualized_log_return = avg_daily_log_return * TRADING_DAYS

    return annualized_log_return

def risk(returns):
    df = returns.copy()
    df = df.pct_change().dropna()

    # Calculate daily return standard deviation and annualize it
    daily_std_dev = df.std()
    annualized_risk = daily_std_dev * np.sqrt(TRADING_DAYS)

    return annualized_risk