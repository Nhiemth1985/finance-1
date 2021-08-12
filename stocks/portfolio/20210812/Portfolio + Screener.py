#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import yfinance as yf
import pypfopt

from pandas_datareader import DataReader
from pypfopt import plotting
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import date
from dateutil.relativedelta import relativedelta
from yahooquery import Screener

start_date = date.today() - relativedelta(months = 12)
end_date = date.today()

s = Screener()
data = s.get_screeners('undervalued_growth_stocks', count = 100)
data_unsorted = pd.DataFrame(data['undervalued_growth_stocks']['quotes'])
data_sorted = data_unsorted.sort_values(by='averageAnalystRating', ascending=True)
data_ssorted = data_sorted.dropna(subset=['averageAnalystRating'])
data_sssorted = data_ssorted.query('averageAnalystRating.str.contains("Buy")',engine='python')
df = pd.DataFrame(data_sssorted,columns = ['symbol'])
assets = df.values.tolist()

data = pd.DataFrame()
for asset in assets:
  data[asset] = DataReader(asset, 'yahoo', start_date, end_date)['Close']

mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)
latest_prices = get_latest_prices(data)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value = 3132.54)
allocation, leftover = da.lp_portfolio()
allocation2, leftover2 = da.greedy_portfolio()

print("INTEGER:")
print("Discrete allocation", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
print("GREEDY:")
print("Discrete allocation", allocation2)
print("Funds remaining: ${:.2f}".format(leftover2))

#pypfopt.plotting.plot_covariance(S)

