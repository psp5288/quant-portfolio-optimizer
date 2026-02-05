import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import yfinance as yf

# Configuration
STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM"]  # You can change these
START_DATE = "2020-01-01"
END_DATE = datetime.datetime.today().strftime("%Y-%m-%d")


def get_data(stocks, start, end):
    print(f"Downloading data for: {stocks}")
    data = yf.download(stocks, start=start, end=end, group_by="column")

    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(0)
        if "Adj Close" in fields:
            data = data["Adj Close"]
        elif "Close" in fields:
            data = data["Close"]
        else:
            raise ValueError("Expected 'Adj Close' or 'Close' in download data.")
    else:
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(axis=1, how="all")
    return data


def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    # 1. Calculate Portfolio Return
    returns = np.sum(mean_returns * weights) * 252

    # 2. Calculate Portfolio Volatility (Standard Deviation)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    return returns, std


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    # Minimize negative Sharpe since optimizer minimizes.
    p_ret, p_var = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def maximize_sharpe_ratio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    # Constraints: The sum of weights must equal 1 (100% allocation)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)

    # Bounds: Each stock weight must be between 0 and 1 (No short selling)
    bound = (0.0, 1.0)
    bounds = tuple(bound for _ in range(num_assets))

    # Initial Guess: Equal distribution
    result = optimize.minimize(
        negative_sharpe,
        num_assets * [1.0 / num_assets],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


if __name__ == "__main__":
    df = get_data(STOCKS, START_DATE, END_DATE)
    tickers = list(df.columns)
    if not tickers:
        raise ValueError("No price data downloaded for the requested tickers.")

    returns = df.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    print("Optimizing portfolio...")
    optimal_result = maximize_sharpe_ratio(mean_returns, cov_matrix)

    optimal_weights = optimal_result.x
    opt_return, opt_volatility = calculate_portfolio_performance(
        optimal_weights, mean_returns, cov_matrix
    )

    print("\n-------------------------------------------")
    print("OPTIMAL PORTFOLIO WEIGHTS:")
    print("-------------------------------------------")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")

    print("-------------------------------------------")
    print(f"Expected Annual Return: {opt_return:.4f} ({opt_return*100:.2f}%)")
    print(
        f"Annual Volatility (Risk): {opt_volatility:.4f} ({opt_volatility*100:.2f}%)"
    )
    print(f"Sharpe Ratio: {(opt_return - 0.01) / opt_volatility:.4f}")

    print("\nGenerating Efficient Frontier Plot...")

    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        p_return, p_std = calculate_portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        results[0, i] = p_std
        results[1, i] = p_return
        results[2, i] = (p_return - 0.01) / p_std

    plt.figure(figsize=(10, 7))
    plt.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="YlGnBu",
        marker="o",
        s=10,
        alpha=0.3,
    )
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(
        opt_volatility,
        opt_return,
        marker="*",
        color="r",
        s=500,
        label="Maximum Sharpe Ratio",
    )
    plt.title("Portfolio Optimization: Efficient Frontier")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Return")
    plt.legend(labelspacing=0.8)
    plt.show()
