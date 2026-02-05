import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide")

st.title("⚡ FlashGen Quantitative Optimizer")
st.markdown("### Modern Portfolio Theory (MPT) Implementation")

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Input")

# Default tickers
default_tickers = "AAPL, MSFT, GOOG, AMZN, TSLA, JPM"
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma separated)", default_tickers)

# Parse tickers
tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- CACHING DATA ---
@st.cache_data
def get_data(tickers, start, end):
    try:
        # Fetch User Stocks
        data = yf.download(tickers, start=start, end=end, group_by="column")
        if isinstance(data.columns, pd.MultiIndex):
            fields = data.columns.get_level_values(0)
            if "Adj Close" in fields:
                data = data["Adj Close"]
            elif "Close" in fields:
                data = data["Close"]
            else:
                st.error("Expected 'Adj Close' or 'Close' in download data.")
                return None
        else:
            if "Adj Close" in data.columns:
                data = data["Adj Close"]
            elif "Close" in data.columns:
                data = data["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.dropna(axis=1, how="all")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data
def get_benchmark(start, end):
    try:
        # Fetch S&P 500 (SPY) for comparison
        spy = yf.download("SPY", start=start, end=end, group_by="column")
        if isinstance(spy, pd.DataFrame) and "Adj Close" in spy.columns:
            spy = spy["Adj Close"]
        elif isinstance(spy, pd.DataFrame) and "Close" in spy.columns:
            spy = spy["Close"]
        if isinstance(spy, pd.DataFrame):
            spy = spy.squeeze()
        return spy
    except Exception as e:
        return None

# --- OPTIMIZATION FUNCTIONS ---
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    p_ret, p_var = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def maximize_sharpe_ratio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    result = optimize.minimize(negative_sharpe, num_assets*[1./num_assets], args=args,
                               method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# --- EXECUTION ---
if st.sidebar.button("Optimize Portfolio"):
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    with st.spinner("Fetching data and optimizing..."):
        # 1. Get Data
        df = get_data(tickers, start_date, end_date)
        
        if df is not None and not df.empty:
            active_tickers = list(df.columns)
            # Display Raw Data Check
            st.expander("Show Raw Data").write(df.tail())

            # Calculations
            returns = df.pct_change()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            # Optimize
            optimal_result = maximize_sharpe_ratio(mean_returns, cov_matrix)
            optimal_weights = optimal_result.x
            opt_return, opt_volatility = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix)

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Optimal Weights")
                weights_df = pd.DataFrame({
                    "Stock": active_tickers,
                    "Weight": optimal_weights
                })
                weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                st.table(weights_df)

            with col2:
                st.subheader("Performance Metrics")
                st.metric("Expected Annual Return", f"{opt_return:.2%}")
                st.metric("Annual Volatility (Risk)", f"{opt_volatility:.2%}")
                st.metric("Sharpe Ratio", f"{(opt_return - 0.02) / opt_volatility:.2f}")

            # --- EFFICIENT FRONTIER ---
            st.markdown("---")
            st.subheader("Efficient Frontier Visualization")
            
            num_portfolios = 5000
            results = np.zeros((3, num_portfolios))
            for i in range(num_portfolios):
                weights = np.random.random(len(active_tickers))
                weights /= np.sum(weights)
                p_return, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
                results[0,i] = p_std
                results[1,i] = p_return
                results[2,i] = (p_return - 0.02) / p_std

            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
            plt.colorbar(sc, label='Sharpe Ratio')
            ax.scatter(opt_volatility, opt_return, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
            ax.set_xlabel('Volatility (Risk)')
            ax.set_ylabel('Return')
            ax.legend()
            st.pyplot(fig)

            # --- BENCHMARK COMPARISON (The New Feature) ---
            st.markdown("---")
            st.subheader("💰 Portfolio vs. S&P 500 Benchmark")
            
            with st.spinner("Comparing against the market..."):
                spy_data = get_benchmark(start_date, end_date)
                
                if spy_data is not None:
                    # Normalize both to start at $100
                    # Portfolio Cumulative Returns
                    weighted_returns = (returns * optimal_weights).sum(axis=1)
                    port_cum_ret = (1 + weighted_returns).cumprod() * 100
                    
                    # SPY Cumulative Returns
                    spy_returns = spy_data.pct_change().dropna()
                    spy_cum_ret = (1 + spy_returns).cumprod() * 100
                    
                    # Create Comparison DataFrame
                    comp_df = pd.DataFrame({
                        "Optimized Portfolio": port_cum_ret,
                        "S&P 500 (SPY)": spy_cum_ret
                    })
                    comp_df = comp_df.dropna() # Handle differing start dates
                    
                    st.line_chart(comp_df)
                    
                    # Calculate final return
                    port_final = port_cum_ret.iloc[-1]
                    spy_final = spy_cum_ret.iloc[-1]
                    
                    if port_final > spy_final:
                        st.success(f"Result: Your optimized strategy beat the market by {port_final - spy_final:.2f} points!")
                    else:
                        st.warning("Result: The market performed better this time.")
                else:
                    st.warning("Could not fetch S&P 500 data for comparison.")
            
        else:
            st.error("No data found. Please check ticker symbols.")
