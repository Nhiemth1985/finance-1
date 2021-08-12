#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules

import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import date
from dateutil.relativedelta import relativedelta


# In[2]:


# set dates

start_date = date.today() + relativedelta(months = -12)
end_date = date.today()


# In[3]:


assets = "F AAPL"


# In[4]:


# download data

data = yf.download(assets, start = start_date, end = end_date)['Close']


# In[5]:


# calculate expected returns and sample covariance

mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# optimize for maximal Sharpe ratio

ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[6]:


# optimization

latest_prices = get_latest_prices(data)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value = 1000)

allocation, leftover = da.lp_portfolio()
allocation2, leftover2 = da.greedy_portfolio()

print("Integer")
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
print("")
print("Greedy")
print("Discrete allocation:", allocation2)
print("Funds remaining: ${:.2f}".format(leftover2))

