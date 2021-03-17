import pandas as pd
import yfinance as yf
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd  
import datetime as dt
import streamlit as st

class Efficient_Frontier:
    
    def __init__(self, stock_df, tickers):
        
        #what is inputted
        self.stock_df = stock_df
        self.tickers = tickers
    
    '''    
    def returns_calculation(self):

        returns_method_list = ['historical_average', "bullish", "bearish", "statistics", "LSTM"]

        if self.returns_method == "historical_average":

            returns = self.stock_df.pct_change().mean()
            return returns

        if self.returns_method == "bullish":

            print("still working on bullish")

        if self.returns_method == "bearish":

            print("still working on bearish")

        if self.returns_method == "statistics":

            print("still working on statistics")

        if self.returns_method == "LSTM":

            print("still working on LSTM")

        else:
            print("incorrect returns method, use any below:")
            for i in returns_method_list:
                print(i)

    def risk_calculation(self):

        risk_measure_list = ["covaraince"]

        if self.risk_measure == "covariance":

            risk = self.stock_df.pct_change().cov()

            return risk
        else:

            print("incorret risk method, use any below")
            for i in risk_measure_list:
                print(i)
    '''
    
    def calc_portfolio_perf(self, weights, mean_returns, cov, rf):
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - rf) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio

    def simulate_random_portfolios(self, num_portfolios, mean_returns, cov, rf, tickers):
        
        results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
        
        for i in range(num_portfolios):
            
            if i % 10000 == 0:
                
                st.write("simulated", '{:,}'.format(i) , "portfolios") 
            
            
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            
            portfolio_return, portfolio_std, sharpe_ratio = self.calc_portfolio_perf(weights, mean_returns, cov, rf)
            
            results_matrix[0,i] = portfolio_return
            results_matrix[1,i] = portfolio_std
            results_matrix[2,i] = sharpe_ratio
            
            for j in range(len(weights)):
                
                results_matrix[j+3,i] = weights[j]
        
        tickers = tickers.split(",")
        results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
        
        return results_df  

        
    def find_portfolios(self, results):
        
        max_sharpe_port = results.iloc[results['sharpe'].idxmax()]
        min_vol_port = results.iloc[results['stdev'].idxmin()]
        
        return max_sharpe_port, min_vol_port

returns_methods = ["means_returns"]
risk_methods = ["covariance"]

st.header("Efficient Frontier")
tickers = st.text_input('Please enter tickers here (seperated by comma):')
status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))

today = dt.date.today()

before = today - dt.timedelta(days=700)
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
    
    ef = Efficient_Frontier(df, tickers)
    
    returns_method = st.selectbox("Select returns method", returns_methods)
    risk_measure = st.selectbox("Select risk method", risk_methods)
    num_portfolios_resp = st.number_input('Please enter number of simulations', min_value = 0, max_value = 1000000, step = 1)
    
    returns_method = df.pct_change().mean()
    risk_measure = df.pct_change().cov()
    num_portfolios = num_portfolios_resp
    rf = 0.0
    
    frontier_radio = st.radio('Please click Search when you are ready to run.', ('Entry', 'Search'))
    if frontier_radio == "Search":
        
        results_frame = ef.simulate_random_portfolios(num_portfolios, returns_method, risk_measure, rf, tickers)   
        
        portfolios = ef.find_portfolios(results_frame)
        
        fig = plt.figure()
        plt.scatter(results_frame['stdev'], results_frame['ret'], c = results_frame.sharpe, cmap = 'RdYlBu')
        
        plt.scatter(portfolios[0][1], portfolios[0][0], marker=(5,1,0),color='r',s=500)
        plt.scatter(portfolios[1][1], portfolios[1][0], marker=(5,1,0),color='g',s=500)
        plt.colorbar()
        plt.xlabel("standard deviation")
        plt.ylabel("return")
        st.pyplot(fig)
        
        st.write("maximum sharpe portfolio", portfolios[0])
        st.write("mininum variance portfolio", portfolios[1]) 

    
    
    