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
objective = cp.Maximize(cp.sum(quantities * np.array([future_prices[stock] for stock in tickers])))
problem = cp.Problem(objective, constraints)
problem.solve()

decisions = []
for i, stock in enumerate(tickers):
    if optimal_quantities[i] > portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0]:
        decisions.append('Buy')
    elif optimal_quantities[i] < portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0]:
        decisions.append('Sell')
    else:
        decisions.append('Hold')

portfolio['Current Price'] = [yf.Ticker(stock).history(period='1d')['Close'][0] for stock in tickers]
portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
portfolio['Optimal Quantity'] = optimal_quantities
portfolio['Decision'] = decisions


plt.figure(figsize=(10, 6))
plt.pie(portfolio['Current Value'], labels=portfolio['Stock'], autopct='%1.1f%%')
plt.title('Current Portfolio Distribution')
plt.show()

portfolio.to_csv('updated_portfolio.csv', index=False)
files.download('updated_portfolio.csv')

print(portfolio)
