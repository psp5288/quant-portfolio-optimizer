# ⚡ FlashGen Quantitative Optimizer

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![SciPy](https://img.shields.io/badge/SciPy-optimization-8CAAE6)

### 📊 Project Overview
FlashGen is a Python-based financial modeling tool designed to solve the asset allocation problem using **Modern Portfolio Theory (MPT)**. It mathematically optimizes stock portfolios to maximize the **Sharpe Ratio** (risk-adjusted return) and proves performance against the **S&P 500 benchmark**.

### 🛠️ Tech Stack
* **Python:** Core logic and statistical processing.
* **SciPy (SLSQP):** Non-linear constrained optimization algorithm.
* **Streamlit:** Interactive frontend for real-time backtesting.
* **Pandas/NumPy:** Vectorized financial data manipulation.
* **Matplotlib:** Data visualization (Efficient Frontier & Equity Curves).

### 🚀 Key Features
* **Optimization Engine:** Calculates optimal asset weights to maximize Sharpe Ratio.
* **Monte Carlo Simulation:** Generates 5,000+ random portfolios to visualize the risk/reward landscape.
* **Live Benchmarking:** Compares the optimized strategy against the S&P 500 (SPY) in real-time.
