# ⚡ FlashGen Quantitative Portfolio Optimizer

### 🚀 Live Demo: [https://parin-quant-portfolio-optimizer.streamlit.app](https://parin-quant-portfolio-optimizer.streamlit.app)

## 📊 Overview
FlashGen is a quantitative financial tool that implements **Modern Portfolio Theory (MPT)** to mathematically determine the ideal asset allocation for a given set of stocks. [cite_start]It focuses on maximizing the **Sharpe Ratio** to find the optimal balance between risk and return[cite: 17, 19, 21].

## 🛠️ Tech Stack
* **Python:** Core data processing and logic.
* [cite_start]**SciPy (SLSQP):** Utilized for non-linear constrained optimization to solve for optimal weights[cite: 17].
* [cite_start]**Streamlit:** Built an interactive web dashboard for real-time portfolio analysis[cite: 12].
* [cite_start]**yFinance API:** Integrated for fetching historical "Adjusted Close" price data[cite: 13].
* [cite_start]**Matplotlib:** Generated the **Efficient Frontier** visualization via Monte Carlo simulations.

## 📈 Key Achievements
* [cite_start]**Optimized Returns:** Identified a portfolio (weighted heavily in GOOG and TSLA) that achieved a **39.23% expected annual return**[cite: 17, 19].
* [cite_start]**Risk Management:** Managed an annual volatility of **33.08%**, resulting in a strong **Sharpe Ratio of 1.13**[cite: 21, 23].
* [cite_start]**Market Outperformance:** Successfully backtested the strategy against the **S&P 500 (SPY)**, outperforming the benchmark by **544.78 points** over the tested period[cite: 63, 64].

## 🧮 How it Works
1. [cite_start]**Data Collection:** The app pulls historical data for user-defined tickers (e.g., AAPL, MSFT, TSLA)[cite: 3].
2. [cite_start]**Statistical Modeling:** It calculates daily returns, mean returns, and the covariance matrix to assess asset correlations[cite: 13].
3. [cite_start]**Optimization Engine:** The `SLSQP` solver minimizes the negative Sharpe Ratio subject to the constraint that all weights must sum to 100%[cite: 17].
4. [cite_start]**Visualization:** - **Efficient Frontier:** Plots thousands of random portfolios to show the risk-reward boundary.
   - [cite_start]**Equity Curve:** Compares the cumulative growth of the optimized portfolio against the S&P 500.
