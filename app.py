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
    tickers = st.text_input("Stocks", "RELIANCE.NS,TCS.NS,HDFCBANK.NS")

with col2:
    weights_input = st.text_input("Weights", "0.4,0.3,0.3")

tickers = [t.strip() for t in tickers.split(",")]
weights = [float(w) for w in weights_input.split(",")]

# -------------------------
# DATA
# -------------------------
data = yf.download(tickers, start="2020-01-01")["Close"]
returns = data.pct_change().dropna()
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
sharpe = (portfolio_returns.mean() - rf) / portfolio_returns.std()

market = yf.download("^NSEI", start="2020-01-01")["Close"]
market_returns = market.pct_change().dropna()

combined = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
combined.columns = ["portfolio", "market"]

beta = combined["portfolio"].cov(combined["market"]) / combined["market"].var()

# -------------------------
# METRIC CARDS
# -------------------------
st.markdown("## 📈 Key Metrics")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hist VaR", f"{var_hist:.4f}")
c2.metric("Param VaR", f"{var_param:.4f}")
c3.metric("MC VaR", f"{var_mc:.4f}")
c4.metric("Sharpe", f"{sharpe:.2f}")
c5.metric("Beta", f"{beta:.2f}")

# -------------------------
# CHART 1 — CUMULATIVE RETURNS
# -------------------------
st.markdown("## 📊 Cumulative Returns")

cum_returns = (1 + portfolio_returns).cumprod()

fig1, ax1 = plt.subplots()
ax1.plot(cum_returns)
ax1.set_title("Portfolio Growth Over Time")
st.pyplot(fig1)

# -------------------------
# CHART 2 — RETURNS DISTRIBUTION
# -------------------------
st.markdown("## 📉 Risk Distribution (VaR Highlighted)")

fig2, ax2 = plt.subplots()
sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax2)
ax2.axvline(var_hist, linestyle="--")
ax2.set_title("Returns Distribution")
st.pyplot(fig2)

# -------------------------
# CHART 3 — CORRELATION HEATMAP
# -------------------------
st.markdown("## 🔗 Correlation Heatmap")

corr = returns.corr()

fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax3)
st.pyplot(fig3)

# -------------------------
# CHART 4 — ROLLING VOLATILITY
# -------------------------
st.markdown("## 📊 Rolling Volatility (Risk Over Time)")

rolling_vol = portfolio_returns.rolling(window=30).std()

fig4, ax4 = plt.subplots()
ax4.plot(rolling_vol)
ax4.set_title("30-Day Rolling Volatility")
st.pyplot(fig4)

# -------------------------
# CHART 5 — DRAWDOWN
# -------------------------
st.markdown("## 📉 Drawdown Analysis")

cum_max = cum_returns.cummax()
drawdown = (cum_returns - cum_max) / cum_max

fig5, ax5 = plt.subplots()
ax5.plot(drawdown)
ax5.set_title("Portfolio Drawdown")
st.pyplot(fig5)