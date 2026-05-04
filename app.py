# -------------------------
# DATA (FIXED)
# -------------------------
try:
    data = yf.download(tickers, start="2020-01-01", progress=False)

    if data.empty:
        st.error("❌ Failed to fetch data. Try different stocks or check internet.")
        st.stop()

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data["Close"]

    returns = data.pct_change().dropna()

    if returns.empty:
        st.error("❌ No returns data available.")
        st.stop()

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()
