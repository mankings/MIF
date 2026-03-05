# MIF - Mercados e Instrumentos Financeiros

Portfolio optimization tool implementing **Modern Portfolio Theory (Markowitz Efficient Frontier)** to find optimal asset allocations that maximize risk-adjusted returns or minimize risk.

## What it does

1. **Fetches historical price data** (2020–2023) for two sector-specific ETF portfolios using Yahoo Finance:
   - **Wallet 1** (Life Sciences): WOOD, CUT, IBB, ERTH
   - **Wallet 2** (Chemicals/Materials): VAW, IYM, DOW, CTVA, MOS, CF, UAN
2. **Computes annualized returns and risk** (volatility) for each asset using log returns over 252 trading days.
3. **Simulates 50,000 random portfolios** to map the efficient frontier and identify the maximum Sharpe ratio portfolio.
4. **Optimizes portfolio weights** using constrained optimization (SciPy SLSQP) for two strategies:
   - **Max Sharpe Ratio** — best risk-adjusted return
   - **Min Risk** — global minimum variance portfolio
5. **Generates plots** — normalized price charts and efficient frontier scatter plots saved to `plots/`.

## Project structure

```
main.py              # Entry point — runs the full workflow
src/
  helpers.py         # Data loading/fetching (yfinance + CSV cache)
  wallets.py         # Normalized price chart visualization
  base_calcs.py      # Return rate and risk calculations
  markowitz.py       # Efficient frontier simulation & optimization
plots/               # Generated charts (PNG)
data/                # Cached price data (CSV)
```

## Dependencies

- pandas, numpy — data handling and math
- yfinance — historical stock data
- matplotlib — plotting
- scipy — portfolio optimization

## Usage

```bash
python main.py
```

This fetches data (or loads from cache), prints optimal weights per ticker, and saves plots to `plots/`.
