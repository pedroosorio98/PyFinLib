import pandas as pd
import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
from pandas_datareader import data as dt
import yfinance as yf
import quandl as qdl
import pandas_datareader as pdr

def sharpe_ratio(rets,rf=0,periods_per_year=252):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        vol = annualized_vol(rets,periods_per_year)
        ret = annualized_rets(rets,periods_per_year)
        return (ret-rf)/vol
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def portfolio_sharpe_ratio(weights,expected_returns,cov_matrix,rf=0,periods_per_year=252):
    return((weights.T @ expected_returns)-rf)/portfolio_vol(weights,cov_matrix,periods_per_year)

def cummulated_returns(rets):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return rets.add(1).prod()-1
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def kurtosis(rets): 
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return ((rets-rets.mean())**4).mean()/(rets.std(ddof=0)**4)
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def skewness(rets): 
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return ((rets-rets.mean())**3).mean()/(rets.std(ddof=0)**3)
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
    
def annualized_rets(rets, periods_per_year=252):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return (rets.add(1).prod())**(periods_per_year/len(rets))-1
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def performance_index(rets): 
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return rets.add(1).cumprod()
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
              
def underwater(rets):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        performance = performance_index(rets)
        return performance/performance.cummax()-1
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def plot_underwater(rets):
    underwater(rets).plot.area(figsize=(16,6))
    plt.legend()

def max_drawdown(rets):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return underwater(rets).min()
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")

def annualized_vol(rets,periods_per_year=252):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        return rets.std()*(periods_per_year**0.5)
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def historical_var(rets,pct=0.05):
     if isinstance(rets, pd.DataFrame):
        return rets.aggregate(historical_var, pct=pct)
     elif isinstance(rets, pd.Series):
        return np.percentile(rets, pct*100)
     else:
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")

def historical_cvar(rets,pct=0.05):
    if isinstance(rets, pd.DataFrame):
        return rets.aggregate(historical_cvar, pct=pct)
    elif isinstance(rets, pd.Series):
        return rets[rets < historical_var(rets)].mean()
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")

def gaussian_var(rets, pct=0.05):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        z = scipy.stats.norm.ppf(pct)
        return rets.mean() + z*rets.std(ddof=0)
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def modified_var(rets, pct=0.05):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        z = scipy.stats.norm.ppf(pct)
        s = skewness(rets)
        k = kurtosis(rets)
        z = (z + (z**2 - 1)*s/6 +(z**3 -3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36)
        return rets.mean() + z*rets.std(ddof=0)
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")

def normality(rets,significance_level=0.05):
    if isinstance(rets,pd.Series):
        stat, p_value = scipy.stats.jarque_bera(rets)
        return p_value>significance_level
    elif isinstance(rets,pd.DataFrame):
        return rets.aggregate(normality,significance_level=significance_level)
    
def main_stats(rets, periods_per_year=252, pct=0.05, significance_level=0.05,rf=0):
    if isinstance(rets,pd.Series) or isinstance(rets,pd.DataFrame):
        
        stats = {"Annualized Return":annualized_rets(rets,periods_per_year=periods_per_year), 
                 "Annualized Volatility":annualized_vol(rets,periods_per_year=periods_per_year),
                 "Sharpe Ratio (rf={0})".format(rf): sharpe_ratio(rets,rf=rf,periods_per_year=periods_per_year),
                 "Maximum Drawdown":max_drawdown(rets),"Historical VaR ({0}%)".format(pct*100):historical_var(rets),
                 "Historical CVaR ({0}%)".format(pct*100):historical_cvar(rets,pct=pct),
                 "Gaussian VaR ({0}%)".format(pct*100):gaussian_var(rets,pct=pct),                                                                            "Modified VaR ({0}%)".format(pct*100):modified_var(rets,pct=pct),"Kurtosis":kurtosis(rets),"Skewness":skewness(rets),
                 "Normality":normality(rets,significance_level=significance_level)}
        if isinstance(rets,pd.DataFrame):
            return pd.DataFrame(stats)
        else:
            return pd.DataFrame(stats,index=["stats"])
    else: 
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def portfolio_vol(weights,cov_matrix,periods_per_year=252):
    return np.sqrt(np.dot(np.dot(weights.T,cov_matrix),weights))*np.sqrt(periods_per_year)

def target_ret_min_vol(cov_matrix,expected_returns, target):
    n = len(expected_returns); guess = np.repeat(1/n,n); weights_bounds = ((0,1),)*n
    return_is_target = {'type':'eq','args':(expected_returns,), 'fun': lambda expected_returns, x : x.T @ expected_returns - target}
    sum_is_one = {'type':'eq','fun':lambda x:np.sum(x)-1}
      
    weights = scipy.optimize.minimize(portfolio_vol, guess, args = (cov_matrix,), method = "SLSQP", options = {"disp": False},
              constraints = (return_is_target, sum_is_one), bounds = weights_bounds).x
    return weights

def plot_markowitz_ef(portfolio_target_rets,cov_matrix,expected_returns):
    weights = [target_ret_min_vol(cov_matrix,expected_returns,i) for i in portfolio_target_rets]
    variance = [portfolio_vol(i,cov_matrix)**2 for i in weights]
    returns = [i.T @ expected_returns for i in weights]

    plt.figure(figsize=(16,6))
    plt.plot(variance,returns)
    plt.title("Markowitz Efficient Frontier",fontsize=14)
    plt.xlabel("Portfolio Variance",fontsize=13)
    plt.ylabel("Portfolio Return",fontsize=13);
    
def msr_portfolio_weights(cov_matrix,expected_returns,rf):
    n = len(expected_returns); guess = np.repeat(1/n,n); weights_bounds = ((0,1),)*n
 
    sum_is_one = {'type':'eq','fun':lambda w : np.sum(w)-1}
    
    def portfolio_negative_sr(weights,expected_returns,cov_matrix, rf):
        portfolio_return = weights.T @ expected_returns; portfolio_volatility = portfolio_vol(weights,cov_matrix)
        return -(portfolio_return-rf)/portfolio_volatility
    
    weights = scipy.optimize.minimize(portfolio_negative_sr, guess, args = (expected_returns,cov_matrix,rf,), method = "SLSQP", options =       {"disp": False}, constraints = (sum_is_one), bounds = weights_bounds).x
    
    return weights

def gmv_portfolio_weights(cov_matrix):
    n = len(cov_matrix); guess = np.repeat(1/n,n); weights_bounds = ((0,1),)*n;expected_returns = np.repeat(1/n,n)
    sum_is_one = {'type':'eq','fun':lambda x:np.sum(x)-1}
      
    weights = scipy.optimize.minimize(portfolio_vol, guess, args = (cov_matrix,), method = "SLSQP", options = {"disp": False},
              constraints = (sum_is_one), bounds = weights_bounds).x
    return weights

def portfolio_annualized_stats(weights,cov_matrix,expected_returns,rf=0,name=0):
    stats = {"Annualized Return": (weights.T @ expected_returns), 
             "Annualized Volatility":portfolio_vol(weights,cov_matrix),
             "Sharpe Ratio":portfolio_sharpe_ratio(weights,expected_returns,cov_matrix,rf)}
    return pd.DataFrame(stats,index=[name])

def no_reb_strategy(assets_returns, initiall_weights):
    if isinstance(assets_returns,pd.DataFrame):
        assets_performance = performance_index(assets_returns)
        return (initiall_weights*(assets_performance/assets_performance.iloc[0])).sum(axis=1)
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame object.")
        
def contribution_to_risk(weights,cov_matrix):
    portfolio_variance = np.dot(np.dot(weights.T,cov_matrix),weights)
    return ((weights*cov_matrix).sum(axis=1)*weights)/portfolio_variance

def rp_portfolio_weights(cov_matrix):
    n = len(cov_matrix); guess = np.repeat(1/n,n); weights_bounds = ((0,1),)*n; rp_weights=ew_weights(n)
    sum_is_one = {'type':'eq','fun':lambda x:np.sum(x)-1}
    
    def rp_func(weights,cov_matrix):
        return ((contribution_to_risk(weights,cov_matrix)-rp_weights)**2).sum()
      
    weights = scipy.optimize.minimize(rp_func, guess, args = (cov_matrix,), method = "SLSQP", options = {"disp": False},
              constraints = (sum_is_one), bounds = weights_bounds).x
    return weights

def ew_weights(n_assets):
    if isinstance(n_assets,int) or isinstance(n_assets,float):
        return np.repeat(1/n_assets,n_assets)
    else:
        raise TypeError("This function is supposed to receive a int or float object.")
    
def get_yf_prices(tickers,start_date=None,end_date=None):
    if isinstance(tickers,list):
        df = pd.DataFrame()
        for ticker in tickers: 
            try:
                df[ticker] = dt.DataReader(ticker,data_source='yahoo',start=start_date,end=end_date)["Adj Close"]
            except:
                print("No data found for: " + str(ticker))    
        return df 
    else: 
        raise TypeError("This function is supposed to receive a list object.") 
        
        
def download_yf_prices(tickers,start_date=None,end_date=None):
    if isinstance(tickers,list):
        df = pd.DataFrame()
        for ticker in tickers: 
            try:
                df[ticker] = yf.download(ticker,start=start_date,end=end_date)["Adj Close"]
            except:
                print("No data found for: " + str(ticker))
        return df
    else: 
        raise TypeError("This function is supposed to receive a list object.") 
        
def get_cdi_bcb(start_date=None,end_date=None):
    return (qdl.get('BCB/12',start_date=start_date,end_date=end_date)/100).rename(columns={'Value':'CDI'})

def get_imab5_bcb(start_date=None,end_date=None):
    return (qdl.get('BCB/12467',start_date=start_date,end_date=end_date)/100).rename(columns={'Value':'IMA-B5'})
                
def get_imab_bcb(start_date=None,end_date=None):
    return (qdl.get('BCB/12466',start_date=start_date,end_date=end_date)/100).rename(columns={'Value':'IMA-B'})

def get_imag_bcb(start_date=None,end_date=None):
    return (qdl.get('BCB/12469',start_date=start_date,end_date=end_date)/100).rename(columns={'Value':'IMA-G'})

def portfolio_return(weights, rets):
    return weights.T @ rets

def rets_yearly_returns(rets):
    if isinstance(rets,pd.DataFrame) or isinstance(rets,pd.Series):
        return ((rets.groupby(rets.index.year).apply(cummulated_returns))*100).round(2)
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame object.")
        
def get_nefin_data():
    return pd.read_csv("PyFinLib Data/NEFIN Data.csv",index_col=0,parse_dates=True)

def get_fammafrench_names():
    return pdr.famafrench.get_available_datasets()

def get_fammafrench_data(data_name,start_date=None,end_date=None):
    if isinstance(data_name,str):
        return dt.DataReader(data_name,data_source="famafrench",start=start_date,end=end_date)
    
def plot_montecarlo_ef(cov_matrix,expected_returns,n_simulations=1000):
    n_assets = len(cov_matrix)
    rand_floats = pd.DataFrame(np.random.rand(n_simulations,n_assets))
    rand_weights = rand_floats.div(rand_floats.sum(axis=1),axis=0)
    
    v = []
    r = []
    for i in rand_weights.index:
        v.append(portfolio_vol(np.array(rand_weights.iloc[i]),cov_matrix)**2)
        r.append(portfolio_return(np.array(rand_weights.iloc[i]),expected_returns))
        
    plt.figure(figsize=(16,6))
    plt.scatter(v,r)
    
def expand_weights(rets,reb_weights): 
    if isinstance(rets,pd.DataFrame) and isinstance(reb_weights,pd.DataFrame):
        daily_weights = rets.copy()
        daily_weights[:]=0
        daily_weights.loc[reb_weights.index] = reb_weights
        previous_day = daily_weights.index[0]
        for i in daily_weights.index[1:]: 
            if all(daily_weights.loc[i]==0):
                amount = ((daily_weights.loc[previous_day].T@rets.loc[i])+1)
                daily_weights.loc[i] = daily_weights.loc[previous_day]*(rets.loc[i].add(1)).div(amount)
            previous_day = i
        return daily_weights
    else:
         raise TypeError("This function is supposed to receive two pd.DataFrame objects.")
            
def momentum_weighted_weights(rets,gamma,look_back_days=252,reb_days=30):
    if isinstance(rets,pd.DataFrame):
        momentum_signal = np.exp(gamma*performance_index(rets).pct_change(look_back_days).dropna(how="all").iloc[::reb_days])
        reb_weights = momentum_signal.div(momentum_signal.sum(axis=1),axis=0)
        return expand_weights(rets,reb_weights).iloc[look_back_days:]
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def strategy_returns(rets,weights):
    rets = rets.loc[weights.index[0]:]
    return rets.mul(weights.shift(1),axis=1).sum(axis=1)

def vol_weighted_weights(rets,look_back_days=252,reb_days=30):
    if isinstance(rets,pd.DataFrame):
        vol_signal = (rets.rolling(look_back_days).std().dropna(how="all").iloc[::reb_days])**(-1)
        reb_weights = vol_signal.div(vol_signal.sum(axis=1),axis=0)
        return expand_weights(rets,reb_weights).iloc[look_back_days:]
    else:
        raise TypeError("This function is supposed to receive a pd.DataFrame or pd.Series object.")
        
def get_fedfunds(start_date=None,end_date=None):
    return qdl.get('FRED/FEDFUNDS',start_date=start_date,end_date=end_date)