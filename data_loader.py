import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats


# ---------------------------
# 1. Load Data
# ---------------------------
def load_data(tickers, start="2020-01-01"):
    data = yf.download(tickers, start=start)

    print("\nAvailable columns:\n", data.columns)

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            data = data["Adj Close"]
        else:
            data = data["Close"]
    else:
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        else:
            data = data["Close"]

    return data


# ---------------------------
# 2. Calculate Returns
# ---------------------------
def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns


# ---------------------------
# 3. Portfolio Returns
# ---------------------------
def calculate_portfolio_returns(returns, weights):
    portfolio_returns = returns.dot(weights)
    return portfolio_returns


# ---------------------------
# 4. Historical VaR
# ---------------------------
def calculate_var_historical(portfolio_returns, confidence_level=0.95):
    return portfolio_returns.quantile(1 - confidence_level)


# ---------------------------
# 5. Parametric VaR
# ---------------------------
def calculate_var_parametric(portfolio_returns, confidence_level=0.95):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    z = stats.norm.ppf(1 - confidence_level)
    return mean + z * std


# ---------------------------
# 6. Monte Carlo VaR
# ---------------------------
def calculate_var_monte_carlo(portfolio_returns, confidence_level=0.95, simulations=10000):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()

    simulated_returns = np.random.normal(mean, std, simulations)
    return pd.Series(simulated_returns).quantile(1 - confidence_level)


# ---------------------------
# 7. Sharpe Ratio
# ---------------------------
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.06):
    daily_rf = risk_free_rate / 252
    excess_returns = portfolio_returns - daily_rf
    return excess_returns.mean() / portfolio_returns.std()


# ---------------------------
# 8. Beta (vs Nifty 50)
# ---------------------------
def calculate_beta(portfolio_returns):
    market = yf.download("^NSEI", start="2020-01-01")["Close"]
    market_returns = market.pct_change().dropna()

    combined = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    combined.columns = ["portfolio", "market"]

    cov = combined["portfolio"].cov(combined["market"])
    var = combined["market"].var()

    return cov / var


# ---------------------------
# MAIN PROGRAM
# ---------------------------
if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

    print("Downloading data...")
    data = load_data(tickers)

    print("\nPrice Data (Last 5 rows):")
    print(data.tail())

    returns = calculate_returns(data)

    print("\nReturns Data (First 5 rows):")
    print(returns.head())

    weights = [0.4, 0.3, 0.3]

    portfolio_returns = calculate_portfolio_returns(returns, weights)

    print("\nPortfolio Returns (First 5 rows):")
    print(portfolio_returns.head())

    # VaR Calculations
    var_hist = calculate_var_historical(portfolio_returns)
    print("\nHistorical VaR (95%):", var_hist)

    var_param = calculate_var_parametric(portfolio_returns)
    print("Parametric VaR (95%):", var_param)

    var_mc = calculate_var_monte_carlo(portfolio_returns)
    print("Monte Carlo VaR (95%):", var_mc)

    # Performance Metrics
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    print("\nSharpe Ratio:", sharpe)

    beta = calculate_beta(portfolio_returns)
    print("Portfolio Beta (vs Nifty 50):", beta)
    
