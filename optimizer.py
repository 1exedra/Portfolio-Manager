import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from google.colab import files

portfolio = pd.DataFrame({
    'Stock': ['AAPL', 'MSFT', 'GOOGL'],
    'Quantity': [10, 5, 2],
    'Purchase Price': [150, 200, 1800]
})

tickers = portfolio['Stock'].tolist()
data = yf.download(tickers, start="2022-01-01", end="2023-12-31")
closing_prices = data['Adj Close']

def prepare_data(stock_prices):
    stock_prices = stock_prices.dropna()
    X = np.array(range(len(stock_prices))).reshape(-1, 1)
    y = stock_prices.values
    return X, y

models = {}
for stock in tickers:
    X, y = prepare_data(closing_prices[stock])
    model = LinearRegression().fit(X, y)
    models[stock] = model
    
future_prices = {}
for stock in tickers:
    days_ahead = 30
    future_day = np.array([[len(closing_prices)]])
    future_price = models[stock].predict(future_day)[0]
    future_prices[stock] = future_price

quantities = cp.Variable(len(tickers), integer=True)

constraints = [
    quantities >= 0,
    cp.sum(quantities * np.array([future_prices[stock] for stock in tickers])) <= sum(portfolio['Quantity'] * portfolio['Purchase Price'])
]
