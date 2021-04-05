import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt
import pandas_datareader as web

from streamlit_efficient_frontier import *

returns_methods = ["mean_returns", "projected_prices"]
risk_methods = ["covariance"]
risk_free_rates = ["4_week_treasury_bill", "3_month_treasury_bill", "6_month_treasury_bill", "1_year_treasury_bill", "manual_input"]

st.header("Efficient Frontier")
with st.beta_expander('purpose'):
    st.write("One of my interests is in portfolio management and optimization, so I've built in a \
             framework for building efficient frontiers, and plan to add more. Such as rebalancing \
                 multiple returns and risk methods, and optimization techniques. Some of the tools in \
                     this application will be backed by research and experimentation on my github")
                     
tickers = st.text_input("Please enter tickers here (seperate):")
st.text("ex for Microsoft, Apple, Amazon, Google, Facebook would be 'MSFT, AAPL, AMZN, GOOG, FB'")
status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))

today = dt.date.today()

before = today - dt.timedelta(days=3653)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

df = pd.DataFrame()
find_frontier = False

if status_radio == "Search":

    df = yf.download(tickers, start_date, end_date)['Adj Close']
    st.dataframe(df)
    
    find_frontier = True


if find_frontier == True:
    
    st.subheader('Options for calculating returns')
    with st.beta_expander("mean returns"):
        ("this gets the daily percent change and then gets the average of that distribution")  
    with st.beta_expander("projected prices"):
        ("plug in projected stock prices for each stock and then optimize to those returns")

    returns_method = st.selectbox("Select returns method", returns_methods)
    
    if returns_method == "projected_prices":
        
        tickers = tickers.split(",")
        tickers = [x.strip(' ') for x in tickers]
            
        st.write("Enter target price and years projected for each company")
        
        projected_prices = pd.DataFrame(columns = ['target', 'years', 'current_price', 'daily_returns(%)'], index = tickers)
        
        for i in tickers:
            
            projected_col1, projected_col2, projected_col3 = st.beta_columns(3)
            
            with projected_col1:
                st.write(yf.Ticker(i).info.get("longName"))
            
            with projected_col2:
                prices = st.number_input("%s target price" % i, min_value = 0, max_value = 1000000)
                projected_prices['target'][str(i)] = prices
                
            with projected_col3:
                year_projection = st.number_input("%s projected years" % i, min_value = 1, max_value = 1000000, step = 1)
                projected_prices['years'] [str(i)] = year_projection
                
        projected_price_radio = st.radio("Click run once dataframe is filled", ("stop", "run"))
        
        if projected_price_radio == "run":

            for i in range(len(projected_prices)):
                
                projected_prices['current_price'][tickers[i]] = yf.Ticker(tickers[i]).info.get('ask')
                projected_prices['daily_returns(%)'][tickers[i]] = np.round(((projected_prices['target'][tickers[i]] - projected_prices['current_price'][tickers[i]]) / projected_prices['current_price'][i]) / (projected_prices['years'][tickers[i]] * 252) * 100,8)
                
            
            st.write(projected_prices)
            projected_returns = projected_prices['daily_returns(%)']
    
    st.subheader("Options for risk measures")
    with st.beta_expander("covariance"):
        ("this creates a covariance matrix of the daily percentage changes of each stock")
    
    risk_measure = st.selectbox("Select risk method", risk_methods)
    num_portfolios_resp = st.number_input('Please enter number of simulations', min_value = 0, max_value = 1000000, step = 1)
    
    rf_end = dt.datetime.today()
    rf_start = rf_end - dt.timedelta(days = 10)
    
    rf_input = st.selectbox("select risk free rate", risk_free_rates)
    
    if rf_input == "4_week_treasury_bill":
        
        rf = web.DataReader('DTB4WK', 'fred', rf_start, rf_end)
        rf = rf.iloc[:, 0][len(rf) - 1]
        st.write("4 week treasury bill:", '{:,}'.format(rf),"%")
        
    if rf_input == "3_month_treasury_bill":
        
        rf = web.DataReader('DTB3', 'fred', rf_start, rf_end)
        rf = rf.iloc[:, 0][len(rf) - 1]
        st.write("3 month treasury bill:", '{:,}'.format(rf),"%")
        
    if rf_input == "6_month_treasury_bill":
        
        rf = web.DataReader('DTB6', 'fred', rf_start, rf_end)
        rf = rf.iloc[:, 0][len(rf) - 1]
        st.write("6 month treasury bill:", '{:,}'.format(rf),"%")
    
    if rf_input == "1_year_treasury_bill":
        
        rf = web.DataReader('DTB1YR', 'fred', rf_start, rf_end)
        rf = rf.iloc[:, 0][len(rf) - 1]
        st.write("1 year treasury bill:", '{:,}'.format(rf),"%")
        
    if rf_input == "manual_input":
        
        rf = st.number_input("enter a risk free rate")
        st.write("manual_input")
    
    
    frontier_radio = st.radio('Please click Search when you are ready to run.', ('Entry', 'Search'))
    if frontier_radio == "Search":
        
        if returns_method == "mean_returns":
            ef = Efficient_Frontier(df, tickers)
        
        if returns_method == "projected_prices":
            ef = Efficient_Frontier(df, tickers, projected_returns)   
        
        results_frame = ef.simulate_random_portfolios(returns_method, risk_measure, num_portfolios_resp, rf, tickers)
        portfolios = ef.find_portfolios(results_frame)
        
        fig = plt.figure()
        plt.scatter(results_frame['stdev'], results_frame['ret'], c = results_frame.sharpe, cmap = 'RdYlBu')
        
        plt.scatter(portfolios[0][1], portfolios[0][0], marker=(5,1,0),color='r',s=500)
        plt.scatter(portfolios[1][1], portfolios[1][0], marker=(5,1,0),color='g',s=500)
        plt.xlabel("standard deviation")
        plt.ylabel("return")
        st.pyplot(fig)
        
        col1, col2 = st.beta_columns(2)
        
        with col1:
            st.write("maximum sharpe portfolio", portfolios[0])
            
        with col2:
            st.write("mininum variance portfolio", portfolios[1]) 

st.write('Disclaimer: Information and output provided on this site does \
         not constitute investment advice.')
st.write('Created by Diego Alvarez')
