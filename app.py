import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Portfolio Risk Analyzer", layout="wide")

st.title("📊 Portfolio Risk Analyzer")
st.markdown("### Quantitative Risk & Performance Dashboard")

# -------------------------
# INPUT
# -------------------------
col1, col2 = st.columns(2)

with col1:
    tickers_input = st.text_input("Stocks", "AAPL,MSFT,GOOGL")

with col2:
    weights_input = st.text_input("Weights", "0.4,0.3,0.3")

# Convert inputs safely
try:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    weights = [float(w) for w in weights_input.split(",")]
except:
    st.error("Invalid input format")
    st.stop()

if len(weights) != len(tickers):
    st.error("⚠️ Number of weights must match number of stocks")
    st.stop()

# -------------------------
# DATA LOADING
# -------------------------
with st.spinner("Fetching market data..."):
    try:
        data = yf.download(tickers, start="2020-01-01", progress=False)

        if data.empty:
            st.error("❌ No data fetched. Try AAPL, MSFT, GOOGL")
            st.stop()

        if isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        else:
            data = data["Close"]

        returns = data.pct_change().dropna()

        if returns.empty:
            st.error("❌ No returns data")
            st.stop()

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# -------------------------
# PORTFOLIO RETURNS
# -------------------------
portfolio_returns = returns.dot(weights)

# -------------------------
# METRICS
# -------------------------
confidence = 0.95

var_hist = portfolio_returns.quantile(1 - confidence)

mean = portfolio_returns.mean()
std = portfolio_returns.std()

z = stats.norm.ppf(1 - confidence)
var_param = mean + z * std

simulated = np.random.normal(mean, std, 10000)
var_mc = pd.Series(simulated).quantile(1 - confidence)

rf = 0.06 / 252
sharpe = (mean - rf) / std if std != 0 else 0

# Beta
try:
    market = yf.download("^GSPC", start="2020-01-01", progress=False)["Close"]
    market_returns = market.pct_change().dropna()

    combined = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    combined.columns = ["portfolio", "market"]

    beta = combined["portfolio"].cov(combined["market"]) / combined["market"].var()
except:
    beta = np.nan

# -------------------------
# DISPLAY
# -------------------------
st.markdown("## 📈 Key Metrics")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Hist VaR", f"{var_hist:.4f}")
c2.metric("Param VaR", f"{var_param:.4f}")
c3.metric("MC VaR", f"{var_mc:.4f}")
c4.metric("Sharpe", f"{sharpe:.2f}")
c5.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A")

# -------------------------
# CHARTS
# -------------------------
st.markdown("## 📊 Cumulative Returns")
cum_returns = (1 + portfolio_returns).cumprod()

fig1, ax1 = plt.subplots()
ax1.plot(cum_returns)
st.pyplot(fig1)

st.markdown("## 📉 Distribution")

fig2, ax2 = plt.subplots()
sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax2)
ax2.axvline(var_hist, linestyle="--")
st.pyplot(fig2)

st.markdown("## 🔗 Correlation")

fig3, ax3 = plt.subplots()
sns.heatmap(returns.corr(), annot=True, ax=ax3)
st.pyplot(fig3)

st.markdown("## 📊 Volatility")

rolling_vol = portfolio_returns.rolling(30).std()

fig4, ax4 = plt.subplots()
ax4.plot(rolling_vol)
st.pyplot(fig4)

st.markdown("## 📉 Drawdown")

cum_max = cum_returns.cummax()
drawdown = (cum_returns - cum_max) / cum_max

fig5, ax5 = plt.subplots()
ax5.plot(drawdown)
st.pyplot(fig5)
