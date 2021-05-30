import numpy as np
import pandas as pd  
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import norm

class VaR:
    
    def __init__(self, df):
        
        self.df = df
        self.returns = self.df.pct_change()
        self.cov_matrix = self.returns.cov()
        
    def ef_var(self, min_var_weights, max_sharpe_weights):
        
        min_var_avg_rets = self.returns.mean()
        min_var_portfolio_mean = min_var_avg_rets @ min_var_weights
        min_var_portfolio_std = np.sqrt(min_var_weights.T @ self.cov_matrix @ min_var_weights)
        min_var_std_investment = 100000 * min_var_portfolio_std

        max_sharpe_avg_rets = self.returns.mean()
        max_sharpe_portfolio_mean = max_sharpe_avg_rets @ max_sharpe_weights
        max_sharpe_portfolio_std = np.sqrt(max_sharpe_weights.T @ self.cov_matrix @ max_sharpe_weights)
        max_sharpe_std_investment = 100000 * max_sharpe_portfolio_std
        
        x = np.arange(-0.05, 0.055, 0.001)
        
        min_var_norm_dist = norm.pdf(x, min_var_portfolio_mean, min_var_portfolio_std)
        max_sharpe_norm_dist = norm.pdf(x, max_sharpe_portfolio_mean, max_sharpe_portfolio_std)
        
        fig1 = plt.figure(figsize = (6,6))
        plt.plot(x, min_var_norm_dist, color='g', label = "Mininum Variance")
        plt.plot(x, max_sharpe_norm_dist, color='r', label = "Maximum Sharpe")
        plt.legend()
        plt.xlabel("Returns (%)")
        plt.ylabel("Frequency")
        plt.title("1 Day VaR returns distribution")
        plt.grid(True)
        
        min_var = norm.ppf(0.05, min_var_portfolio_mean, min_var_portfolio_std)
        max_sharpe = norm.ppf(0.05, max_sharpe_portfolio_mean, max_sharpe_portfolio_std)
        
        min_var_mean_investment = 100000 * (1 + min_var_portfolio_mean)
        min_var_std_investment = 100000 * min_var_portfolio_std
        
        max_sharpe_mean_investment = 100000 * (1 + max_sharpe_portfolio_mean)
        max_sharpe_std_investment = 100000 * max_sharpe_portfolio_std
        
        min_var_cutoff = norm.ppf(0.05, min_var_mean_investment, min_var_std_investment)
        max_sharpe_cutoff = norm.ppf(0.05, max_sharpe_mean_investment, max_sharpe_std_investment)
        
        min_var_historical_var = 100000 - min_var_cutoff
        max_sharpe_historical_var = 100000 - max_sharpe_cutoff
        
        min_var_array = []
        max_sharpe_array = []
        
        num_days = int(15)
        for x in range(1, num_days + 1):
            
            min_var_array.append(np.round(min_var_historical_var * np.sqrt(x),2))
            max_sharpe_array.append(np.round(max_sharpe_historical_var * np.sqrt(x),2))
        
        fig2 = plt.figure(figsize = (6,6))
        plt.xlabel("Day")
        plt.ylabel("Max portfolio loss (USD)")
        plt.title("Max portfolio loss (VaR) over 15-day period")
        plt.plot(min_var_array, "g", label = "mininum variance")
        plt.plot(max_sharpe_array, "r", label = "maximum variance")    
        plt.legend()
        plt.grid(True)
        
        col1, col2 = st.beta_columns(2)
        
        with col1:
            
            st.pyplot(fig1)
            st.subheader("Mininum Variance Portfolio")
            st.write("mininum variance portfolio expected daily return: {}%".format(round(min_var_portfolio_mean * 100, 4)))
            st.write("mininum variance portfolio daily volatility: {}".format(round(min_var_portfolio_std, 6)))
            st.write("1 day mininum variance VaR with 95% confidence: {}%".format(round((100 * min_var), 3)))
            st.write("mininum variance cutoff value: ${:,}".format(round(min_var_cutoff, 2)))
            st.write("mininum variance historical VaR: ${:,}".format(round(min_var_historical_var, 2)))
            
        with col2:
            
            st.pyplot(fig2)
            st.subheader("Maximum Sharpe Portfolio")
            st.write("maximum sharpe portfolio expected daily return: {}%".format(round(max_sharpe_portfolio_mean * 100, 4)))
            st.write("maximum sharpe portfolio daily volatility: {}".format(round(max_sharpe_portfolio_std, 6)))
            st.write("1 day maximum sharpe VaR with 95% confidence: {}%".format(round((100 * max_sharpe), 3)))
            st.write("maximum sharpe cutoff value: ${:,}".format(round(max_sharpe_cutoff,2)))
            st.write("maximum sharpe historical VaR: ${:,}".format(round(max_sharpe_historical_var, 2)))
            
        st.write("_____________________________________________")
        
        
    def standard_var(self, weights):
        
        st.write("Assumming ${:,} portfolio".format(100000))
        
        avg_rets = self.returns.mean()
        portfolio_mean = avg_rets @ weights
        portfolio_std = np.sqrt(weights.T @ self.cov_matrix @ weights)
        std_investment = 100000 * portfolio_std
        
        x = np.arange(-0.05, 0.055, 0.001)
        norm_dist = norm.pdf(x, portfolio_mean, portfolio_std)
        
        fig1 = plt.figure(figsize = (6,6))
        plt.xlabel("Returns (%)")
        plt.ylabel("Frequency")
        plt.plot(x, norm_dist)
        plt.title("1 Day VaR returns distribution")
        plt.grid(True)
        
        var = norm.ppf(0.05, portfolio_mean, portfolio_std)
        mean_investment = 100000 * (1 + portfolio_mean)
        std_investment = 100000 * portfolio_std
        cutoff = norm.ppf(0.05, mean_investment, std_investment)
        historical_var = 100000 - cutoff
        
        array = []
        num_days = int(15)
        
        for x in range(1, num_days + 1):
            array.append(np.round(historical_var * np.sqrt(x), 2))
            
        fig2 = plt.figure(figsize = (6,6))
        plt.xlabel("Day")
        plt.ylabel("Max portfolio loss (USD)")
        plt.title("Max portfolio loss (VaR) over 15-day period")
        plt.plot(array) 
        plt.grid(True)
        
        col1, col2 = st.beta_columns(2)
        
        with col1: 
            st.pyplot(fig1)
            
        with col2:
            st.pyplot(fig2)
            
        st.write("expected daily return: {}%".format(round(portfolio_mean * 100, 4)))
        st.write("expected daily volatility {}".format(round(portfolio_std, 6)))
        st.write("1 day VaR with 95% confidence: {}%".format(round((100 * var), 3)))
        st.write("cutoff value: ${:,}".format(round(cutoff, 2)))
        st.write("historical VaR: ${:,}".format(round(historical_var, 2)))
        st.write("_____________________________________________")
            
        
        
        