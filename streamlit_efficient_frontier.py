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
        

