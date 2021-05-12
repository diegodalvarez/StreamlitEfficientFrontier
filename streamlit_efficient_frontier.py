import numpy as np
import pandas as pd  
import datetime as dt
import streamlit as st

class Efficient_Frontier:
    
    def __init__(self, stock_df, tickers, *projected_returns):
        
        #what is inputted
        self.stock_df = stock_df
        self.tickers = tickers
        self.projected_returns = pd.DataFrame(projected_returns).transpose().squeeze()
    
    def returns_calculation(self, method, *projected_returns):
        
        if method == "mean_returns":
            
            returns = self.stock_df.pct_change().mean()
            return returns
        
        if method == "projected_prices":
            return self.projected_returns
        
        if method == "log_returns":

            returns = self.stock_df.pct_change()
            for i in self.stock_df.columns:
                
                returns[i] = np.log(1 + returns[i])

            returns = returns.mean()
            return returns     
        
    def risk_calculation(self, method):
        
        if method == "covariance" or method == "cov":
            
            risk = self.stock_df.pct_change().cov()
            return risk

    def calc_portfolio_perf(self, weights, mean_returns, cov, rf):
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - rf) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio

    def simulate_random_portfolios(self, returns_method, risk_method, num_portfolios, rf, tickers):
        
        returns = self.returns_calculation(returns_method)
        risk = self.risk_calculation(risk_method)
        
        results_matrix = np.zeros((len(returns)+3, num_portfolios))
        
        for i in range(num_portfolios):
            
            if i % 10000 == 0:
        
                st.write("simulated", '{:,}'.format(i) , "portfolios") 
            
            weights = np.random.random(len(returns))
            weights /= np.sum(weights)
            
            portfolio_return, portfolio_std, sharpe_ratio = self.calc_portfolio_perf(weights, returns, risk, rf)
            
            results_matrix[0,i] = portfolio_return
            results_matrix[1,i] = portfolio_std
            results_matrix[2,i] = sharpe_ratio
            
            for j in range(len(weights)):
                results_matrix[j+3,i] = weights[j]       
        
        ticker_type_test = isinstance(tickers, str)
        
        if ticker_type_test == True:
            tickers = tickers.split(',')
            
            
        results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
        
        return results_df  

        
    def find_portfolios(self, results):
        
        max_sharpe_port = results.iloc[results['sharpe'].idxmax()]
        min_vol_port = results.iloc[results['stdev'].idxmin()]
        
        return max_sharpe_port, min_vol_port
