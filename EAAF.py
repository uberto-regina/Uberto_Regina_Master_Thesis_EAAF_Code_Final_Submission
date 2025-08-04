import numpy as np
import pandas as pd
import yfinance as yf
from math import sqrt
from tqdm import tqdm
import random
import cvxpy as cp

from utilities import Portfolio
from sklearn.covariance import LedoitWolf
from pandas_datareader import data as web

from utilities import Portfolios_Allocation

from dateutil.relativedelta import relativedelta
import pickle

import concurrent.futures
from functools import partial
import time


from utilities_plots import compare_weights_score_with_baseline, statistics_comparison_EW

def macro_lagged(): 
    """
    Load and properly lag time‐series data for asset prices, macroeconomic indicators, and the daily risk-free rate.

    This function imports the datasets described in Section 4.1, including:
      - Historical asset price series
      - Macroeconomic indicator time series
      - Daily risk-free rate

    Returns
    -------
    prices : pd.DataFrame
        Date-indexed DataFrame of historical asset prices.
    macroeconomic_indicators : pd.DataFrame
        Date-indexed DataFrame of macroeconomic variables.
    daily_rf_rate : pd.Series or float
        Daily risk-free rate as a time series (or constant if unchanging).

    Please note that because we do not own the datasets, they have not been added to the Datasets folder, so the code will not run as is.
    Before executing the script, update this function so it points to your own data.
    """

    CPI_Urban = pd.read_csv('Datasets/InflationYoY.csv').rename(columns={'observation_date':'Date'})
    CPI_Urban['Date'] = pd.to_datetime(CPI_Urban['Date'])
    CPI_Urban = CPI_Urban.set_index('Date')
    CPI_Urban = CPI_Urban.shift(2).dropna()

    Industrial_Prod = pd.read_csv('Datasets/IndustrialProductionYoY.csv').rename(columns={'observation_date':'Date'})
    Industrial_Prod['Date'] = pd.to_datetime(Industrial_Prod['Date'])
    Industrial_Prod = Industrial_Prod.set_index('Date')
    Industrial_Prod = Industrial_Prod.shift(2).dropna()

    PCEPI = pd.read_csv('Datasets/PCEPIYoY.csv').rename(columns={'observation_date':'Date'})
    PCEPI['Date'] = pd.to_datetime(PCEPI['Date'])
    PCEPI = PCEPI.set_index('Date')
    PCEPI = PCEPI.shift(2).dropna()

    Equity_Vola = pd.read_csv('Datasets/Equity_Vola.csv').rename(columns={'observation_date':'Date'})
    Equity_Vola['Date'] = pd.to_datetime(Equity_Vola['Date'])
    Equity_Vola = Equity_Vola.set_index('Date')
    Equity_Vola = Equity_Vola.shift(1).dropna()
    Equity_Vola = Equity_Vola.resample('MS').ffill()
    Equity_Vola.dropna(inplace=True)

    Y10CM = pd.read_csv('Datasets/10Y3M_Treasury_Spread.csv').rename(columns={'observation_date':'Date'})
    Y10CM['Date'] = pd.to_datetime(Y10CM['Date'])
    Y10CM = Y10CM.set_index('Date')
    Y10CM = Y10CM.shift(1).dropna()
    Y10CM = Y10CM.resample('MS').ffill()

    BAA = pd.read_csv('Datasets/DBAA.csv').rename(columns={'observation_date':'Date'})
    BAA['Date'] = pd.to_datetime(BAA['Date'])
    BAA = BAA.set_index('Date')
    BAA = BAA.resample('MS').ffill()

    GS20 = pd.read_csv('Datasets/GS20.csv').rename(columns={'observation_date':'Date'})
    GS20['Date'] = pd.to_datetime(GS20['Date'])
    GS20 = GS20.set_index('Date')
    GS20 = GS20.resample('MS').ffill()

    GS30 = pd.read_csv('Datasets/GS30.csv').rename(columns={'observation_date':'Date'})
    GS30['Date'] = pd.to_datetime(GS30['Date'])
    GS30 = GS30.set_index('Date')
    GS30 = GS30.resample('MS').ffill()

    df = pd.concat([BAA, GS20, GS30], axis=1)

    df['T_avg'] = df[['GS20', 'GS30']].mean(axis=1)
    df['CreditSpread'] = df['DBAA'] - df['T_avg']

    Credit_Spread = df[['CreditSpread']].dropna().shift(1).dropna()
    Credit_Spread = Credit_Spread.resample('MS').ffill()

    Conditions_index = pd.read_csv('Datasets/NFCI_weekly.csv').rename(columns={'observation_date':'Date'})
    Conditions_index['Date'] = pd.to_datetime(Conditions_index['Date'])
    Conditions_index = Conditions_index.set_index('Date')
    Conditions_index = Conditions_index.shift(1).dropna()
    Conditions_index = Conditions_index.resample('MS').ffill().dropna()

    GDP_growth = pd.read_csv('Datasets/GDP_Growth_YoY.csv').rename(columns={'observation_date':'Date'})
    GDP_growth['Date'] = pd.to_datetime(GDP_growth['Date'])
    GDP_growth = GDP_growth.set_index('Date')
    GDP_growth = GDP_growth.resample('MS').ffill()
    GDP_growth = GDP_growth.shift(1).dropna()

    Bonds = pd.read_csv('Datasets/US10Yzerocouponyield.csv', on_bad_lines='skip')
    Bonds['Date'] = pd.to_datetime(Bonds['Date'])
    Bonds = Bonds.set_index('Date')
    Bonds = Bonds.dropna()
    Bonds.columns = ['Yield']

    # Resample data to monthly frequency by taking the first observation of each month
    Monthly_bonds = Bonds.resample('MS').ffill()
    # Calculate monthly returns by adjusting the yield exponent factor from 1/52 to 1/12 for a month
    Monthly_bonds['monthly_return'] = (np.exp(- (10 - 1/12) * Monthly_bonds['Yield'] * 0.01) / 
                                        np.exp(-10 * Monthly_bonds['Yield'].shift(1) * 0.01)) - 1
    # Compute the cumulative product of (1 + monthly_return) to obtain the bond index
    Monthly_bonds['Bonds'] = (1 + Monthly_bonds['monthly_return']).cumprod()
    # Create a DataFrame for bond prices
    Bond_price = pd.DataFrame(Monthly_bonds['Bonds'], index=Monthly_bonds.index)

    Commodities = pd.read_csv('Datasets/Commodities.csv').rename(columns={'observation_date':'Date'})
    Commodities['Date'] = pd.to_datetime(Commodities['Date'])
    Commodities = Commodities.set_index('Date')
    Commodities = Commodities.resample('MS').ffill()
    Commodities.dropna(inplace=True)

    sp500 = pd.read_pickle('Datasets/SP500.pkl') #yf.download('^GSPC', start='1950-01-01', end='2025-01-01', auto_adjust=False)
    SP500 = pd.DataFrame({})
    SP500['SP500'] = sp500['Adj Close']['^GSPC']
    SP500 = SP500.resample('MS').ffill()
    SP500.dropna(inplace=True)

    MOM = pd.DataFrame({})
    MOM['MOM_12'] = SP500['SP500'].shift(1)  / SP500['SP500'].shift(12) - 1

    MacroEconomic_indicators = pd.concat([Credit_Spread,
                                          Y10CM,
                                        Conditions_index, Industrial_Prod, 
                                        CPI_Urban, Equity_Vola, GDP_growth, PCEPI, MOM], axis = 1, join = 'inner') 

    MacroEconomic_indicators.columns = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']

    High_Yield = pd.read_excel('Datasets/High_Yield.xlsx').rename(columns = {'Dates' : 'Date', 'PX_LAST': 'HY'})
    High_Yield['Date'] = pd.to_datetime(High_Yield['Date'])
    High_Yield = High_Yield.set_index('Date')
    High_Yield = High_Yield.resample('MS').ffill()
    High_Yield.dropna(inplace=True)

    Credits = pd.read_csv('Datasets/Credits.csv').rename(columns={'observation_date':'Date', 'BAMLCC0A0CMTRIV' : 'Credits'})
    Credits['Date'] = pd.to_datetime(Credits['Date'])
    Credits = Credits.set_index('Date')

    risk_free = pd.read_pickle('Datasets/risk_free.pkl') #web.DataReader("DGS3MO", "fred", '1984-01-03', '2024-12-31')  # % annual yield
    risk_free = risk_free.ffill()  # forward-fill to handle missing data

    # Convert annual yield to daily **return** (decimal). Assuming 252 trading days in a year.
    daily_rf_rate = (1 + risk_free/100) ** (1/252) - 1
    daily_rf_rate.columns = ["rf"]  

    Prices = pd.concat([SP500, Bond_price, Commodities, Credits, High_Yield], axis = 1, join = 'inner')

    common_index = Prices.index.intersection(MacroEconomic_indicators.index)
    MacroEconomic_indicators = MacroEconomic_indicators.loc[common_index]
    Prices = Prices.loc[common_index]

    # Create a union of both indices
    combined_index = daily_rf_rate.index.union(Prices.index)
    daily_rf_rate = daily_rf_rate.reindex(combined_index).ffill()

    return MacroEconomic_indicators, Prices, daily_rf_rate


class Node:
    def __init__(self, depth=0):
        self.depth = depth
        self.feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.score = None
        self.is_leaf = False
        self.weights = None

class EAATree:
    """
    Core decision-tree building block (EAAT) of the EAAF framework for dynamic asset allocation.

    This class encapsulates an Enhanced-Asset Allocation Tree, which recursively partitions 
    macroeconomic data to generate asset allocation rules tailored to changing market conditions.

    Parameters
    ----------
    Prices : pd.DataFrame
        Date-indexed historical price series for all assets in the universe.
    assets : list of str
        Names of the assets to include in the tree.
    data : pd.DataFrame
        Date-indexed macroeconomic indicators used as explanatory variables.
    daily_rf : float or pd.Series
        Daily risk-free rate for target computation and excess-return adjustments.
    compute_target_fct : callable
        Function that computes the target variable (Eq. 3.7.) for splits.
    max_depth : int, optional
        Maximum depth of the tree (default is 4).
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node, equal to twice the minimum number of samples per leaf (default is 72).
    lambda_s : float, optional
        Penalty parameter for deviation from benchmark(default is 0).
    lambda_tc : float, optional
        Penalty parameter for transaction costs in allocation (default is 0).
    prev_w_t_1 : np.array, optional
        Previous period's asset weights for continuity between rebalances (default is None).
    prev_day : datetime-like, optional
        Date of the previous rebalancing period (default is None).
    seed : int, optional
        Random seed for reproducible splits (default is 42).
    """

    def __init__(self, Prices, assets, data, daily_rf, compute_target_fct, max_depth=4, min_samples_split=72, lambda_s = 0, lambda_tc = 0, prev_w_t_1 = None, prev_day = None, seed = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.assets = assets
        self.Prices_memory = Prices.copy()
        common_index = Prices[self.assets].index.intersection(data.index)
        self.compute_target = compute_target_fct
        self.previous_weights = prev_w_t_1
        self.prev_day = prev_day


        self.Prices = Prices[self.assets] 
        self.data = data.loc[common_index]
        self.daily_rf_rate = daily_rf  

        self.lambda_s = lambda_s
        self.lambda_tc = lambda_tc

        self.seed = seed
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)


    def fit(self, features):
        self.features = features
        self.root = self._build_tree(self.data, depth=0)
        
    def _build_tree(self, data, depth):
        node = Node(depth=depth)

        # Stop splitting if maximum depth is reached or too few samples remain.
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            node.is_leaf = True
            node.score, node.weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.prev_day, self.lambda_s, self.lambda_tc)
            return node

        if depth == 0:
            best_score = -np.inf
            old_score = -np.inf
            _, parent_weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.prev_day, self.lambda_s, self.lambda_tc)
            
        else:
            old_score, parent_weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.prev_day, self.lambda_s, self.lambda_tc)

        best_score = -np.inf
        best_feature = None
        best_split = None
        best_left = None
        best_right = None
        best_w_l = None
        best_w_r = None

        # Feature subsampling: consider only a random subset of features
        k = max(1, int(sqrt(len(self.features))))
        features_to_consider = self._py_rng.sample(self.features, k)

        # Count total candidate thresholds across all features with variation
        total_candidates = 0
        for feature in features_to_consider:
            feature_min = data[feature].min()
            feature_max = data[feature].max()
            if feature_min != feature_max:
                values = data[feature].values
                unique_vals = np.unique(np.sort(values))
                all_midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

                unique_mid = np.unique(np.sort(all_midpoints))
                min_val, max_val = unique_mid[0], unique_mid[-1]

                # This creates 20 equidistant thresholds including the endpoints.
                candidate_thresholds = np.linspace(min_val, max_val, 20)   
                total_candidates += max(0, len(candidate_thresholds) - 1)

        with tqdm(total=total_candidates, desc=f"Depth {depth}") as pbar:
            for feature in features_to_consider:
                feature_min = data[feature].min()
                feature_max = data[feature].max()
                if feature_min == feature_max:
                    continue
        
                values = data[feature].values
                unique_vals = np.unique(np.sort(values))
                all_midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

                unique_mid = np.unique(np.sort(all_midpoints))
                min_val, max_val = unique_mid[0], unique_mid[-1]

                # This creates 20 equidistant thresholds including the endpoints.
                candidate_thresholds = np.linspace(min_val, max_val, 20)


                for t in candidate_thresholds:
                    pbar.update(1)
                    left_count = np.sum(data[feature] <= t)
                    right_count = np.sum(data[feature] > t)


                    # Keep only thresholds that yield at least 36 data points on each side.

                    min_threshold = round(self.min_samples_split/2)


                    if left_count < min_threshold or right_count < min_threshold: 
                        continue

                    left = data[data[feature] <= t]
                    right = data[data[feature] > t]

                    
                    # Skip splits that don't partition the data
                    if len(left) == 0 or len(right) == 0:
                        continue


                    score_left, weight_left = self.compute_target(left, self.Prices, self.daily_rf_rate, self.previous_weights, self.prev_day, self.lambda_s, self.lambda_tc)
                    score_right, weight_right = self.compute_target(right, self.Prices, self.daily_rf_rate, self.previous_weights, self.prev_day, self.lambda_s, self.lambda_tc)

                    total_score = ((len(left)*score_left) / (len(left)+len(right))) + ((len(right)*score_right) / (len(right) + len(left)))

                    if total_score > best_score and score_left > old_score and score_right > old_score: 
                        best_score = total_score
                        best_feature = feature
                        best_split = t
                        best_left = left
                        best_right = right
                        best_w_l = weight_left
                        best_w_r = weight_right

        # If no valid split is found, mark the node as a leaf.
        if best_feature is None:
            node.is_leaf = True
            node.score, node.weights = self.compute_target(data, self.Prices, self.daily_rf_rate,  self.previous_weights, self.prev_day, self.lambda_s, self.lambda_tc)
            return node

        # Save the best split information and recursively build child nodes.
        node.feature = best_feature
        node.split_value = best_split
        node.left = self._build_tree(best_left, depth + 1)
        node.right = self._build_tree(best_right, depth + 1)

        return node
    
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.is_leaf:
            print(indent + f"Leaf: score = {node.score:.4f}, weights = ", node.weights)
        else:
            print(indent + f"{node.feature} < {node.split_value:.4f}?")
            self.print_tree(node.left, indent + "  ")
            self.print_tree(node.right, indent + "  ")
    
    
    def _predict_(self,
                          dataset: pd.DataFrame,
                          prev_weights,
                          idx: int,
                          node=None,
                          history: pd.DataFrame = None):
        """
        Recursively traverse the tree for the datapoint at position `idx`,
        carrying forward only those past rows that follow the same splits
        as dataset.iloc[idx].  When we hit a leaf, compute_target on that
        filtered history.
        """
        if node is None:
            node = self.root
            # start with all past up to idx
            history = dataset.iloc[: idx]
        # if we're at a leaf, compute on the filtered history
        if node.is_leaf:
            _, w_opt = self.compute_target(
                history,
                self.Prices_memory,
                self.daily_rf_rate,
                prev_weights,
                dataset.iloc[: idx].iloc[-1].name,
                self.lambda_s,
                self.lambda_tc
            )
            return w_opt

        # otherwise, decide branch based on the single datapoint at idx
        
        x = dataset.iloc[idx]
        if x[node.feature] < node.split_value:
            # filter history to match “went left”
            history_left = history[history[node.feature] < node.split_value]
            return self._predict_(
                dataset,
                prev_weights,
                idx,
                node.left,
                history_left
            )
        else:
            # filter history to match “went right”
            history_right = history[history[node.feature] >= node.split_value]
            return self._predict_(
                dataset,
                prev_weights,
                idx,
                node.right,
                history_right
            )
        
    def predict_sequence(self, dataset: pd.DataFrame, initial_weights):
        """
        Walk through dataset row 0 to row N-1, each time using
        the full history up to that row in compute_target.
        """
        data_full = self.data.copy()
        full_dataset = pd.concat([data_full, dataset], axis = 0)
        preds = []
        prev_w = initial_weights
        for i in range(len(dataset)):
            index = len(data_full) + i
            w_i = self._predict_(full_dataset, prev_w, index)
            preds.append(w_i)
            prev_w = w_i
        preds = pd.DataFrame(preds, index = dataset.index)
        preds.columns = self.assets


        preds[0] = list(zip(preds['SP500'], preds['Bonds'], preds['Commodities'], preds['Credits'], preds['HY']))

        lst = preds[[0]]
        lst = pd.DataFrame(lst).rename(columns={0:'Weights'})
        lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
        lab_encountered = []
        portfolio_list = []
        
        new_prices = self.Prices_memory[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']]
        new_prices = new_prices[new_prices.index.isin(lst.index)]
        assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']

        for t in lst.index:
            if lst['label'].loc[t] not in lab_encountered:
                weights = {}
                for i, asset in enumerate(assets):
                    weights[asset] = lst['Weights'].loc[t][i]

                
                new_portfolio = Portfolio(f"Portfolio {lst['label'].loc[t]}", weights
                                        , assets, new_prices)
                portfolio_list.append(new_portfolio)
                lab_encountered.append(lst['label'].loc[t])

        lst['label'] = "Portfolio " + lst['label'].astype(str)
        prices_used = new_prices

        portfolios = portfolio_list
        allocation = pd.DataFrame(lst['label'])

        PA = Portfolios_Allocation(prices_used, portfolios, allocation, assets)
        
        return PA
    
class EAAForest:
    def __init__(self, Prices, assets, data, daily_rf, compute_target_fct,
                 max_depth=4, min_samples_split=72, 
                 n_trees=360, bootstrap_frac=1, lambda_s = 0, lambda_tc = 0, boot = False,
                 previous_weights = None, prev_day = None, random_state = None):
        """
    Implementation of the EAAF: Ensemble of Enhanced Asset Allocation Trees (EAAT) to form the EAAF for dynamic asset allocation.

    This class builds a forest of EAATs, and aggregates their allocation suggestions to form a more robust portfolio strategy.

    Parameters
    ----------
    Prices : pd.DataFrame
        Date-indexed price series for each asset in the universe.
    assets : list of str
        Names of the assets to include in each tree.
    data : pd.DataFrame
        Date-indexed macroeconomic indicators and other explanatory variables.
    daily_rf : float or pd.Series
        Daily risk-free rate for computing excess returns.
    compute_target_fct : callable
        Function that computes the target variable (Eq. 3.7.) for splits.
    max_depth : int, optional
        Maximum depth of each tree (default is 4).
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node, equal to twice the minimum number of samples per leaf (default is 72).
    n_trees : int, optional
        Number of trees in the forest (default is 360).
    bootstrap_frac : float, optional
        Fraction of the training set to sample (with replacement if boot = True, False otherwise) for each tree (default is 1).
    lambda_s : float, optional
        Penalty parameter for deviation from benchmark(default is 0).
    lambda_tc : float, optional
        Penalty parameter for transaction costs in allocation (default is 0).
    boot : bool, optional
        If True, enable bootstrapping of data for each tree (default is False).
    previous_weights : dict or pd.Series, optional
        Weights from the previous period for continuity between rebalances (default is None).
    prev_day : datetime-like, optional
        Date of the previous rebalance (default is None).
    random_state : int, optional
        Seed for reproducible bootstrap sampling (default is None).
    """
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_trees = n_trees
        self.bootstrap_frac = bootstrap_frac
        self.assets = assets
        self.Prices = Prices
        self.compute_target = compute_target_fct
        self.daily_rf = daily_rf

        self.data = data.copy()
        self.trees = []
        self.previous_weights = previous_weights
        self.prev_day = prev_day

        self.lambda_s = lambda_s
        self.lambda_tc = lambda_tc
        self.boot = boot

        self.random_state = random_state

    def _build_one_tree(self, seed, features):
        """
        Builds a single PortfolioTree without parallel execution.
        """
        # 1) draw a bootstrap sample
        bootstrap_sample = self.data.sample(
            frac=self.bootstrap_frac,
            replace=self.boot,
            random_state=seed
        ).sort_index()

        # 2) init and fit a PortfolioTree on that sample
        tree = EAATree(
            self.Prices,
            self.assets,
            bootstrap_sample,
            self.daily_rf,
            self.compute_target,
            self.max_depth,
            self.min_samples_split,
            self.lambda_s,
            self.lambda_tc,
            self.previous_weights,
            self.prev_day,
            seed=int(seed)
        )
        tree.fit(features)
        return tree

    def fit(self, features):
        """
        Build `n_trees` PortfolioTree models sequentially.
        After completion, `self.trees` will hold all trained trees.
        """
        self.features = features

        # initialize random generator and draw unique seeds
        base_rng = np.random.RandomState(self.random_state)
        max32 = np.iinfo(np.int32).max
        seeds = base_rng.randint(0, max32, size=self.n_trees)

        # clear any existing trees
        self.trees = []

        # sequential build
        for seed in seeds:
            trained_tree = self._build_one_tree(seed, features)
            self.trees.append(trained_tree)

    def _predict_(self, dataset: pd.DataFrame, prev_weights, idx: int):
        """
        Ask each tree for its w_opt at row `idx` (using filtered history up to idx),
        then average across trees.
        """
        tree_preds = []
        for tree in self.trees:
            w_i = tree._predict_(dataset, prev_weights, idx)
            tree_preds.append(w_i)
        return np.mean(tree_preds, axis=0)

    def predict_sequence(self, dataset: pd.DataFrame, initial_weights):
        """
        Walk through dataset row 0 to row N-1, each time using
        the full history up to that row in compute_target.
        """
        data_full = self.data.copy()
        full_dataset = pd.concat([data_full, dataset], axis = 0)
        preds = []
        prev_w = initial_weights
        for i in range(len(dataset)):
            index = len(data_full) + i
            w_i = self._predict_(full_dataset, prev_w, index)
            preds.append(w_i)
            prev_w = w_i
        preds = pd.DataFrame(preds, index = dataset.index)
        preds.columns = self.assets


        preds[0] = list(zip(preds['SP500'], preds['Bonds'], preds['Commodities'], preds['Credits'], preds['HY']))

        lst = preds[[0]]
        lst = pd.DataFrame(lst).rename(columns={0:'Weights'})
        lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
        lab_encountered = []
        portfolio_list = []
        
        new_prices = self.Prices[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']]
        new_prices = new_prices[new_prices.index.isin(lst.index)]
        assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']

        for t in lst.index:
            if lst['label'].loc[t] not in lab_encountered:
                weights = {}
                for i, asset in enumerate(assets):
                    weights[asset] = lst['Weights'].loc[t][i]

                
                new_portfolio = Portfolio(f"Portfolio {lst['label'].loc[t]}", weights
                                        , assets, new_prices)
                portfolio_list.append(new_portfolio)
                lab_encountered.append(lst['label'].loc[t])

        lst['label'] = "Portfolio " + lst['label'].astype(str)
        prices_used = new_prices

        portfolios = portfolio_list
        allocation = pd.DataFrame(lst['label'])

        PA = Portfolios_Allocation(prices_used, portfolios, allocation, assets)
        
        return PA
    
class EAAF_parallelized:
    """
    Parallelized Implementation of the EAAF: Ensemble of Enhanced Asset Allocation Trees (EAAT)
    to form the EAAF for dynamic asset allocation.

    This class builds a forest of EAATs, with parallelization to build up to 45 trees simultaneously
    (adjust `n_jobs` as needed), and aggregates their allocation suggestions to form a more robust
    portfolio strategy.

    Parameters
    ----------
    Prices : pd.DataFrame
        Date-indexed price series for each asset in the universe.
    assets : list of str
        Names of the assets to include in each tree.
    data : pd.DataFrame
        Date-indexed macroeconomic indicators and other explanatory variables.
    daily_rf : float or pd.Series
        Daily risk-free rate for computing excess returns.
    compute_target_fct : callable
        Function that computes the target variable (Eq. 3.7.) for splits.
    max_depth : int, optional
        Maximum depth of each tree (default is 4).
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node, equal to twice the minimum number of samples per leaf (default is 72).
    n_trees : int, optional
        Number of trees in the forest (default is 360).
    bootstrap_frac : float, optional
        Fraction of the training set to sample (with replacement if boot = True, False otherwise) for each tree (default is 1).
    lambda_s : float, optional
        Penalty parameter for deviation from benchmark (default is 0).
    lambda_tc : float, optional
        Penalty parameter for transaction costs in allocation (default is 0).
    boot : bool, optional
        If True, enable bootstrapping of data for each tree (default is False).
    previous_weights : dict or pd.Series, optional
        Weights from the previous period for continuity between rebalances (default is None).
    prev_day : datetime-like, optional
        Date of the previous rebalance (default is None).
    random_state : int, optional
        Seed for reproducibility (default is None).
    n_jobs : int, optional
        Number of parallel jobs for building trees (default is 45, using all cores).
    """

    def __init__(
        self,
        Prices,
        assets,
        data,
        daily_rf,
        compute_target_fct,
        max_depth=4,
        min_samples_split=72,
        n_trees=360,
        bootstrap_frac=1,
        lambda_s = 0,
        lambda_tc = 0,
        boot = False,
        previous_weights=None,
        prev_day=None,
        random_state = None,
        n_jobs = 45,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_trees = n_trees
        self.bootstrap_frac = bootstrap_frac
        self.assets = assets
        self.Prices = Prices
        self.compute_target = compute_target_fct
        self.daily_rf = daily_rf
        self.data = data.copy()
        self.previous_weights = previous_weights
        self.prev_day = prev_day
        self.trees = []

        self.lambda_s = lambda_s
        self.lambda_tc = lambda_tc
        self.boot = boot
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _build_one_tree(self, seed, features):
        """
        Runs inside its own process. We pass a unique seed so that
        the bootstrap sample (and any subsequent random.feature subsampling)
        is different in each worker.
        """
        # 1) draw a bootstrap sample in this worker
        bootstrap_sample = self.data.sample(
            frac=self.bootstrap_frac,
            replace=self.boot,
            random_state=seed
        ).sort_index()

        # 2) init and fit a PortfolioTree on that sample
        tree = EAATree(
            self.Prices,
            self.assets,
            bootstrap_sample,
            self.daily_rf,
            self.compute_target,
            self.max_depth,
            self.min_samples_split,
            self.lambda_s,
            self.lambda_tc,
            self.previous_weights,
            self.prev_day,
            seed = int(seed)
        )
        tree.fit(features)
        return tree

    def fit(self, features):
        """
        Build `n_trees` PortfolioTree models in parallel, using up to 45 CPUs.
        After completion, `self.trees` will hold all trained trees.
        """
        self.features = features

        base_rng = np.random.RandomState(self.random_state)
        max32  = np.iinfo(np.int32).max
        seeds  = base_rng.randint(0, max32, size=self.n_trees)

        self.trees = []

        max_workers = min(self.n_jobs, self.n_trees)
        build_fn = partial(self._build_one_tree, features=features)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Launch one “build tree” job per seed
            futures = [executor.submit(build_fn, seed) for seed in seeds]

            # As soon as each tree is done, append it to self.trees
            for future in concurrent.futures.as_completed(futures):
                trained_tree = future.result()
                self.trees.append(trained_tree)

    def _predict_(self, dataset: pd.DataFrame, prev_weights, idx: int):
        tree_preds = []
        for tree in self.trees:
            w_i = tree._predict_(dataset, prev_weights, idx)
            tree_preds.append(w_i)
        return np.mean(tree_preds, axis=0)

    def predict_sequence(self, dataset: pd.DataFrame, initial_weights):

        data_full = self.data.copy()
        full_dataset = pd.concat([data_full, dataset], axis=0)
        preds = []
        prev_w = initial_weights

        for i in range(len(dataset)):
            index = len(data_full) + i
            w_i = self._predict_(full_dataset, prev_w, index)
            preds.append(w_i)
            prev_w = w_i

        preds = pd.DataFrame(preds, index=dataset.index)
        preds.columns = self.assets
        
        preds[0] = list(zip(preds['SP500'], preds['Bonds'], preds['Commodities'], preds['Credits'], preds['HY']))

        lst = preds[[0]]
        lst = pd.DataFrame(lst).rename(columns={0:'Weights'})
        lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
        lab_encountered = []
        portfolio_list = []
        
        new_prices = self.Prices[['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']]
        new_prices = new_prices[new_prices.index.isin(lst.index)]
        assets = ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY']

        for t in lst.index:
            if lst['label'].loc[t] not in lab_encountered:
                weights = {}
                for i, asset in enumerate(assets):
                    weights[asset] = lst['Weights'].loc[t][i]

                
                new_portfolio = Portfolio(f"Portfolio {lst['label'].loc[t]}", weights
                                        , assets, new_prices)
                portfolio_list.append(new_portfolio)
                lab_encountered.append(lst['label'].loc[t])

        lst['label'] = "Portfolio " + lst['label'].astype(str)
        prices_used = new_prices

        portfolios = portfolio_list
        allocation = pd.DataFrame(lst['label'])

        PA = Portfolios_Allocation(prices_used, portfolios, allocation, assets)
        
        return PA
    
def compute_maxreturn_capped_vol(
    data,
    Prices,
    daily_rf_rate,
    previous_weights=None,
    prev_day=None,
    lambda_s=0.,
    lambda_tc=0.
):
    """
    Leaf-splitting optimization as in Eq. 3.7.
    One-step convex optimization: maximize portfolio expected return subject to
        1) w ≥ 0
        2) sum(w) = 1
        3) portfolio annualized volatility ≤ equally-weighted portfolio's annualized volatility

    Parameters
    ----------
    data : pd.DataFrame
        Date-indexed macroeconomic and explanatory variables.
    Prices : pd.DataFrame
        Date-indexed historical asset prices.
    daily_rf_rate : float or pd.Series
        Daily risk-free rate (kept for API consistency).
    previous_weights : pd.Series, optional
        Last period's weights (for turnover penalty).
    prev_day : scalar, optional
        Date label for realized returns lookup.
    lambda_s : float, optional
        Shrinkage penalty coefficient (not used here).
    lambda_tc : float, optional
        Turnover penalty coefficient (if previous_weights provided).

    Returns
    -------
    tuple
        - float: optimal portfolio expected return (μᵀwₒₚₜ) for use as split score.
        - np.array : a vector containing the optimizal weights.
    """

    # 1) Compute monthly returns aligned with `data` index
    Returns = (
        Prices
        .resample('MS')      # month‐start frequency
        .ffill()
        .pct_change()
        .shift(-1)           # so that return at month t corresponds to period [t, t+1)
        .dropna()
    )

    # 2) If we want realized returns for turnover adjustment:
    if prev_day is not None and previous_weights is not None:
        realized = Returns.loc[prev_day].values

    # 3) Restrict to the same dates as `data.index`
    Returns = Returns.reindex(data.index).dropna().sort_index()
    assets = Returns.columns
    n = len(assets)

    # 4) Compute historical expected returns (sample mean)
    T, p = Returns.shape
    mu_vec = Returns.mean().values

    # 5) Compute LW shrinkage covariance estimate
    lw = LedoitWolf().fit(Returns)
    Sigma = lw.covariance_  # monthly covariance matrix

    EW_weights = np.ones(n)/n 
    var_ew_monthly = EW_weights @ Sigma @ EW_weights

    # 6) Set up previous‐weight drift (if applying turnover penalty)
    if previous_weights is None:
        w_prev_drift = np.ones(n) / n
    else:
        w_prev = previous_weights.copy()
        if prev_day is not None:
            w_prev_drift = w_prev * (1 + realized)
            w_prev_drift = w_prev_drift / w_prev_drift.sum()
        else:
            w_prev_drift = w_prev.copy()

    # 7) Build CVXPY variables
    w = cp.Variable(n, nonneg=True)     # portfolio weights
    # Epigraph variables for turnover penalty
    p = cp.Variable(n, nonneg=True)
    q = cp.Variable(n, nonneg=True)
    # Epigraph variables for shrinkage penalty (if used)
    if lambda_s > 0:
        u = cp.Variable(n, nonneg=True)
        v = cp.Variable(n, nonneg=True)

    # 8) Constraints:
    constraints = []
    # (a) Fully invested
    constraints.append(cp.sum(w) == 1)

    # (b) Volatility cap: w' Σ w ≤ var_ew_monthly
    constraints.append(cp.quad_form(w, Sigma) <= var_ew_monthly)

    # (c) Turnover‐penalty epigraph (only if lambda_tc > 0)
    if lambda_tc > 0:
        constraints.append(p >= w - w_prev_drift)
        constraints.append(q >= w_prev_drift - w)

    # (d) Shrinkage‐penalty epigraph (only if gamma > 0 and shrinkage_alloc is provided)
    if lambda_s > 0:
        w_shrink = EW_weights.copy()
        constraints.append(u >= w - w_shrink)
        constraints.append(v >= w_shrink - w)

    # 9) Objective: maximize expected return − λ_tc * turnover_penalty − γ * shrinkage_penalty
    exp_ret = mu_vec @ w
    if lambda_tc > 0:
        turnover_term = lambda_tc * cp.sum(p + q)
    else:
        turnover_term = 0

    if lambda_s > 0:
        shrinkage_term = lambda_s * cp.sum(u + v)
    else:
        shrinkage_term = 0

    objective = cp.Maximize(exp_ret - turnover_term - shrinkage_term)

    # 10) Solve the QP
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=cp.ECOS,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-8,
            max_iters=5000
        )
    except cp.SolverError as e:
        print(f"SolverError occurred: {e}. Using equal‐weighted weights instead.")
        w_opt = EW_weights.copy()
    else:
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Solver did not succeed. Status = {problem.status}. Using equal‐weighted weights instead.")
            w_opt = EW_weights.copy()
        else:
            w_opt = np.maximum(w.value, 0)
            w_opt = w_opt / np.sum(w_opt)

    return mu_vec @ w_opt, w_opt


def six_month_expanding_window_training(MacroEconomic_indicators, Prices, daily_rf_rate):

    """
    Conduct a backtest using an expanding-window evaluation with 6-month out-of-sample periods for training and testing
    the EAAF model, from June 2002 through June 2024 as explained in Section 4.2.

    Starting at June 1, 2002, this function creates successive non-overlapping six-month
    test intervals (e.g., Jun-Nov 2002, Dec 2002-May 2003, …), and for each interval:
    1. Trains an EAAFParallel model on all data up to the day before the test window.
    2. Generates portfolio allocations over the six-month test span.
    3. Captures the terminal portfolio weights at the end of each test window.

    It supports multiple independent repetitions (default one), returning a list of
    DataFrames—one per run—where each DataFrame holds the terminal weights for each
    six-month interval, labeled by date.

    To modify the parameters of the EAAF, it suffices to manually change them in the definition
    of the model below.

    Parameters
    ----------
    MacroEconomic_indicators : pd.DataFrame
        Date-indexed macroeconomic and explanatory variables used as tree features.
    Prices : pd.DataFrame
        Date-indexed historical prices for the universe of assets.
    daily_rf_rate : float or pd.Series
        Daily risk-free rate series (preserved for consistency but not directly used).

    Returns
    -------
    List[pd.DataFrame]
        A list of length `n_test`, each entry being a DataFrame of terminal weights
        (indexed by date) for every six-month test period in that run.
    """

    features = list(MacroEconomic_indicators.columns)

    test_periods = []
    test_start   = pd.Timestamp('2002-06-01')
    last_date    = pd.Timestamp('2024-06-01') 

    first_last = test_start  - relativedelta(months=1)


    while test_start < last_date:
        test_end = test_start + relativedelta(months=6) - relativedelta(days=1)
        if test_end > last_date:
            test_end = last_date
        test_periods.append((test_start, test_end))
        test_start = test_start + relativedelta(months=6)

    # number of independent repetitions
    n_test = 1

    # container for all runs
    Weights_test = []


    for j in range(n_test):
        print(f"\n=== TEST RUN {j+1}/{n_test} ===")
        Weights= []

        weight_last = None
        last_day = first_last

        for start, end in test_periods:
            train_end = start - relativedelta(days=1)
            print(f" Train up to {train_end.date()} → Test {start.date()}–{end.date()}")

            # slice train/test
            train_macro = MacroEconomic_indicators.loc[:train_end]
            train_rf    = daily_rf_rate.loc[:train_end]
            test_macro  = MacroEconomic_indicators.loc[start:end]
            
            Forest = EAAF_parallelized(
                Prices,
                ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                train_macro,
                train_rf,
                compute_maxreturn_capped_vol,
                max_depth=4,
                min_samples_split=72, 
                n_trees=360,
                bootstrap_frac=1,
                lambda_s = 0.,
                lambda_tc = 0.,     
                boot = False, 
                previous_weights=weight_last,
                prev_day=last_day,
                random_state=42
            )

            Forest.fit(features)

            # one allocation call over the full 6-month block
            PA_6m = Forest.predict_sequence(test_macro, weight_last)
            _, _, _, weight_6m, _ , _= PA_6m.calculate_PnL_transactions()

            Weights.append(weight_6m)

            last_day = Weights[-1].iloc[-1].name
            weight_last = []

            for key in PA_6m.portfolios[-1].weights.keys():
                weight_last.append(PA_6m.portfolios[-1].weights[key])  #We retrieve the final weight to use as the starting point for the next training cycle (i.e., W_{T-})

        Weights_df = pd.concat(Weights, axis=0)

        Weights_df['Name'] = f"experiment_{j}"

        # collect this run’s list of DataFrames
        Weights_test.append(Weights_df)

    return Weights_test


def six_month_fixed_rolling_window_training(MacroEconomic_indicators, Prices, daily_rf_rate):
    """
    Conduct a backtest using a fixed-length rolling-window evaluation with 6-month out-of-sample periods 
    for training and testing the EAAF model, from June 2002 through June 2024 (see Section 4.2).

    At each six-month test interval (e.g. Jun–Nov 2002, Dec 2002–May 2003, …):
    1. Select a fixed look-back window of N years (e.g. 10, 12, 15…) ending the day before the test window.
    2. Train an EAAFParallel model on the data within that window.
    3. Generate portfolio allocations over the six-month test span.
    4. Record the terminal portfolio weights at the end of the test window.

    This routine cycles through a predefined list of look-back lengths (`n_years`), 
    producing one weight‐record DataFrame per look-back period. Each DataFrame contains 
    the terminal weights for every six-month test period, labeled by their test-end dates.

    Parameters
    ----------
    MacroEconomic_indicators : pd.DataFrame
        Date-indexed macroeconomic/explanatory variables used as features.
    Prices : pd.DataFrame
        Date-indexed asset price series for the investable universe.
    daily_rf_rate : float or pd.Series
        Daily risk-free rate series (included for API consistency).

    Returns
    -------
    List[pd.DataFrame]
        One DataFrame per look-back length in `n_years`, each listing the terminal 
        weights (indexed by date) for every six-month test interval in that run.
    """

    features = list(MacroEconomic_indicators.columns)
    n_years = [10,12,15,17,18,20] # set number of years for rolling window

    test_periods = []
    test_start = pd.Timestamp('2002-06-01')
    last_date = pd.Timestamp('2024-06-01')  

    while test_start < last_date:
        test_end = test_start + relativedelta(months=6) - relativedelta(days=1)
        if test_end > last_date:
            test_end = last_date
        test_periods.append((test_start, test_end))
        test_start += relativedelta(months=6)

    Weights_test = []

    for j, n_year in enumerate(n_years):
        print(f"\n=== TEST RUN {j+1}/{n_year} ===")
        Weights = []

        weight_last = None
        last_day = None

        for start, end in test_periods:
            train_start = start - relativedelta(years=n_year)
            train_end = start - relativedelta(days=1)

            print(f" Train {train_start.date()}–{train_end.date()} → Test {start.date()}–{end.date()}")

            # slice train/test sets
            train_macro = MacroEconomic_indicators.loc[train_start:train_end]
            train_rf = daily_rf_rate.loc[train_start:train_end]
            test_macro = MacroEconomic_indicators.loc[start:end]

            Forest = EAAF_parallelized(
                Prices,
                ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                train_macro,
                train_rf,
                compute_maxreturn_capped_vol,
                max_depth=4,
                min_samples_split=72,
                n_trees=360,
                bootstrap_frac=1,
                gamma=0.,
                lambda_tc=0.,
                boot=False,
                previous_weights=weight_last,
                prev_day=last_day,
                random_state=42
            )

            Forest.fit(features)

            PA_6m = Forest.predict_sequence(test_macro, weight_last)
            _, _, _, weight_6m, _ = PA_6m.calculate_PnL_transactions()
            Weights.append(weight_6m)

            last_day = Weights[-1].iloc[-1].name
            weight_last = [PA_6m.portfolios[-1].weights[key] for key in PA_6m.portfolios[-1].weights]

        Weights_df = pd.concat(Weights, axis=0)
        Weights_df['Name'] = f"n_years_{n_year}"

        Weights_test.append(Weights_df)

    return Weights_test

if __name__ == '__main__':
    # Example usage

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = six_month_expanding_window_training(MacroEconomic_indicators, Prices, daily_rf_rate)

    compare_weights_score_with_baseline(WT, Prices, MacroEconomic_indicators, daily_rf_rate)

    statistics_comparison_EW(WT, Prices, MacroEconomic_indicators, daily_rf_rate, names = ['EAAF'])

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")








