import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import sqrt
from tqdm import tqdm
import random

from sklearn.covariance import LedoitWolf
from pandas_datareader import data as web

from dateutil.relativedelta import relativedelta
import pickle

import concurrent.futures
from functools import partial
import time

import os
import sys

# To import files outside of this folder
THIS_DIR = os.path.dirname(__file__)                         
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))  
sys.path.insert(1, PROJECT_ROOT)

from EAAF import macro_lagged
from utilities_plots import compare_weights_score_with_baseline
from utilities import Portfolio, Portfolios_Allocation

"""
Functions to replicate the base AAF model of Bettencourt, L. O., Tetereva, A., & Petukhina, A. (2024). Advancing Markowitz: Asset Allocation Forest. SSRN. https://doi.org/10.2139/ssrn.4781685
"""

# Define a simple node structure
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


class AATree:
    def __init__(self, Prices, assets, data, daily_rf, compute_target_fct, max_depth=3, min_samples_split=10, lambda_tc = 0, prev_w_t_1 = None, seed = None):
        """
        Core decision-tree building block (AAT) of the baseline AAF framework for dynamic asset allocation.

        This class encapsulates a baseline Asset Allocation Tree, which recursively partitions 
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
            Function that computes the target variable for splits.
        max_depth : int, optional
            Maximum depth of the tree (default is 4).
        min_samples_split : int, optional
            Minimum number of samples required to split an internal node, equal to twice the minimum number of samples per leaf (default is 72).
        lambda_tc : float, optional
            Penalty parameter for transaction costs in allocation (default is 0).
        prev_w_t_1 : np.array, optional
            Previous period's asset weights for continuity between rebalances (default is None).
        seed : int, optional
            Random seed for reproducible splits (default is 42).
        """
       
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.assets = assets
        self.Prices_memory = Prices.copy()
        common_index = Prices[self.assets].index.intersection(data.index)
        self.compute_target = compute_target_fct
        self.previous_weights = prev_w_t_1

        self.Prices = Prices[self.assets] 
        self.data = data.loc[common_index] 
        self.daily_rf_rate = daily_rf  

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
            node.score, node.weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.lambda_tc)
            return node

        if depth == 0:
            best_score = -np.inf
            old_score = -np.inf
            _, parent_weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.lambda_tc)
            
        else:
            old_score, parent_weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.lambda_tc)

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

                    min_threshold = round(self.min_samples_split/2)

                    # Keep only thresholds that yield at least min_threshold data points on each side.
                    if left_count < min_threshold or right_count < min_threshold: 
                        continue

                    left = data[data[feature] <= t]
                    right = data[data[feature] > t]

                    
                    # Skip splits that don't partition the data
                    if len(left) == 0 or len(right) == 0:
                        continue


                    score_left, weight_left = self.compute_target(left, self.Prices, self.daily_rf_rate, self.previous_weights, self.lambda_tc)
                    score_right, weight_right = self.compute_target(right, self.Prices, self.daily_rf_rate, self.previous_weights, self.lambda_tc)

                    total_score = (( score_left) / (len(left))) + ((score_right) / (len(right)))

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
            node.score, node.weights = self.compute_target(data, self.Prices, self.daily_rf_rate, self.previous_weights, self.lambda_tc)
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
    
    
    def predict(self, datapoint, node=None):
        if node is None:
            node = self.root
        # If this node is a leaf, return its score
        if node.is_leaf:
            return node.weights
        # Check which branch to follow based on the splitting feature and threshold
        if datapoint[node.feature] < node.split_value:
            return self.predict(datapoint, node.left)
        else:
            return self.predict(datapoint, node.right)
        
    def predict_allocation(self, df, if_alloc = False):
        if not if_alloc:
            return df.apply(lambda row: self.predict(row), axis=1)
        else:
            lst = df.apply(lambda row: self.predict(row), axis=1)
            lst = pd.DataFrame(lst).rename(columns={0:'Weights'})
            lst['label'] = pd.factorize(lst['Weights'].apply(tuple))[0]
            lab_encountered = []
            portfolio_list = []
            new_prices = self.Prices_memory[self.assets]
            new_prices = new_prices[new_prices.index.isin(lst.index)]

            for t in lst.index:
                if lst['label'].loc[t] not in lab_encountered:
                    weights = {}
                    for i, assets in enumerate(self.assets):
                        weights[assets] = lst['Weights'].loc[t][i]
                    new_portfolio = Portfolio(f"Portfolio {lst['label'].loc[t]}", weights
                                            , self.assets, new_prices)
                    portfolio_list.append(new_portfolio)
                    lab_encountered.append(lst['label'].loc[t])

            lst['label'] = "Portfolio " + lst['label'].astype(str)
            prices_used = new_prices
            
            portfolios = portfolio_list
            allocation = pd.DataFrame(lst['label'])
            assets = self.assets

            PA = Portfolios_Allocation(prices_used, portfolios, allocation, assets)

        return PA
    
class AAForest:
    def __init__(self, Prices, assets, data, daily_rf, compute_target_fct,
                 max_depth=3, min_samples_split=10, 
                 n_trees=10, bootstrap_frac=0.8, lambda_tc = 0, boot = True, previous_weights = None, random_state = None):
        """
    Implementation of the baseline AAF: Ensemble of  Asset Allocation Trees (EAAT) to form the AAF for dynamic asset allocation.

    This class builds a forest of AATs, and aggregates their allocation suggestions to form a more robust portfolio strategy.

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
        Function that computes the target variable for splits.
    max_depth : int, optional
        Maximum depth of each tree (default is 4).
    min_samples_split : int, optional
        Minimum number of samples required to split an internal node, equal to twice the minimum number of samples per leaf (default is 72).
    n_trees : int, optional
        Number of trees in the forest (default is 200).
    bootstrap_frac : float, optional
        Fraction of the training set to sample (with replacement if boot = True, False otherwise) for each tree (default is 0.9).
    lambda_tc : float, optional
        Penalty parameter for transaction costs in allocation (default is 0).
    boot : bool, optional
        If True, enable bootstrapping of data for each tree (default is True).
    previous_weights : dict or pd.Series, optional
        Weights from the previous period for continuity between rebalances (default is None).
    random_state : int, optional
        Seed for reproducible bootstrap sampling (default is 42).
    frac: Fraction of data to sample (with replacement) for each tree.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_trees = n_trees
        self.bootstrap_frac = bootstrap_frac
        self.assets = assets
        self.Prices = Prices
        self.compute_target = compute_target_fct
        self.daily_rf = daily_rf
        # Save the original data to sample from
        self.data = data.copy()
        self.trees = []
        self.previous_weights = previous_weights
        self.boot = boot

        self.lambda_tc = lambda_tc

        self.random_state = random_state

    def _build_one_tree(self, seed, features):
        """
        Builds a single AATree without parallel execution.
        """
        # 1) draw a bootstrap sample
        bootstrap_sample = self.data.sample(
            frac=self.bootstrap_frac,
            replace=self.boot,
            random_state=seed
        ).sort_index()

        # 2) init and fit a PortfolioTree on that sample
        tree = AATree(
            self.Prices,
            self.assets,
            bootstrap_sample,
            self.daily_rf,
            self.compute_target,
            self.max_depth,
            self.min_samples_split,
            self.lambda_tc,
            self.previous_weights,
            seed=int(seed)
        )
        tree.fit(features)
        return tree

    def fit(self, features):
        """
        Build `n_trees` AATree models sequentially.
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

    def predict(self, datapoint):
        """
        Predict portfolio weights for a single datapoint by averaging the predictions
        from all trees in the forest.
        """
        predictions = [tree.predict(datapoint) for tree in self.trees]
        predictions = np.array(predictions)
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction

    def predict_allocation(self, df, if_alloc=False):
        """
        For each row in df, aggregate predictions across trees.
        
        if_alloc=False: returns a Series of aggregated weight vectors.
        if_alloc=True: builds portfolios based on unique predictions and returns a 
                       Portfolios_Allocation object.
        """
        # Aggregate predictions for each row (each datapoint)
        aggregated_preds = df.apply(lambda row: self.predict(row), axis=1)
        
        if not if_alloc:
            return aggregated_preds
        else:
            # Create a DataFrame with the aggregated predictions
            pred_df = pd.DataFrame(aggregated_preds, columns=["Weights"])
            # Factorize the unique weight vectors to assign portfolio labels
            pred_df["label"] = pd.factorize(pred_df["Weights"].apply(tuple))[0]
            lab_encountered = []
            portfolio_list = []
            new_prices = self.Prices[self.assets]
            new_prices = new_prices[new_prices.index.isin(pred_df.index)]
            
            for t in pred_df.index:
                if pred_df["label"].loc[t] not in lab_encountered:
                    weights = {asset: pred_df["Weights"].loc[t][i] 
                               for i, asset in enumerate(self.assets)}
                    new_portfolio = Portfolio(f"Portfolio {pred_df['label'].loc[t]}", 
                                              weights, self.assets, new_prices)
                    portfolio_list.append(new_portfolio)
                    lab_encountered.append(pred_df["label"].loc[t])
            
            pred_df["label"] = "Portfolio " + pred_df["label"].astype(str)
            allocation = pd.DataFrame(pred_df["label"])
            portfolios = portfolio_list
            assets = self.assets
            PA = Portfolios_Allocation(new_prices, portfolios, allocation, assets)
            return PA

        return PA
    
class AAF_parallelized:
    def __init__(
        self,
        Prices,
        assets,
        data,
        daily_rf,
        compute_target_fct,
        max_depth=3,
        min_samples_split=10,
        n_trees=10,
        bootstrap_frac=0.8,
        lambda_s = 0,
        lambda_tc = 0,
        boot = False,
        previous_weights=None,
        prev_day=None,
        random_state = None,
        n_workers = 45,
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
        self.n_workers = n_workers

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
        tree = AATree(
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
        Build `n_trees` AATree models in parallel, using up to 45 CPUs.
        After completion, `self.trees` will hold all trained trees.
        """
        self.features = features

        base_rng = np.random.RandomState(self.random_state)
        max32  = np.iinfo(np.int32).max
        seeds  = base_rng.randint(0, max32, size=self.n_trees)

        self.trees = []

        max_workers = min(self.n_workers, self.n_trees)
        build_fn = partial(self._build_one_tree, features=features)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Launch one “build tree” job per seed
            futures = [executor.submit(build_fn, seed) for seed in seeds]

            # As soon as each tree is done, append it to self.trees
            for future in concurrent.futures.as_completed(futures):
                trained_tree = future.result()
                self.trees.append(trained_tree)

    def predict(self, datapoint):
        """
        Predict portfolio weights for a single datapoint by averaging the predictions
        from all trees in the forest.
        """
        predictions = [tree.predict(datapoint) for tree in self.trees]
        predictions = np.array(predictions)
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction

    def predict_allocation(self, df, if_alloc=False):
        """
        For each row in df, aggregate predictions across trees.
        
        if_alloc=False: returns a Series of aggregated weight vectors.
        if_alloc=True: builds portfolios based on unique predictions and returns a 
                       Portfolios_Allocation object.
        """
        # Aggregate predictions for each row (each datapoint)
        aggregated_preds = df.apply(lambda row: self.predict(row), axis=1)
        
        if not if_alloc:
            return aggregated_preds
        else:
            # Create a DataFrame with the aggregated predictions
            pred_df = pd.DataFrame(aggregated_preds, columns=["Weights"])
            # Factorize the unique weight vectors to assign portfolio labels
            pred_df["label"] = pd.factorize(pred_df["Weights"].apply(tuple))[0]
            lab_encountered = []
            portfolio_list = []
            new_prices = self.Prices[self.assets]
            new_prices = new_prices[new_prices.index.isin(pred_df.index)]
            
            for t in pred_df.index:
                if pred_df["label"].loc[t] not in lab_encountered:
                    weights = {asset: pred_df["Weights"].loc[t][i] 
                               for i, asset in enumerate(self.assets)}
                    new_portfolio = Portfolio(f"Portfolio {pred_df['label'].loc[t]}", 
                                              weights, self.assets, new_prices)
                    portfolio_list.append(new_portfolio)
                    lab_encountered.append(pred_df["label"].loc[t])
            
            pred_df["label"] = "Portfolio " + pred_df["label"].astype(str)
            allocation = pd.DataFrame(pred_df["label"])
            portfolios = portfolio_list
            assets = self.assets
            PA = Portfolios_Allocation(new_prices, portfolios, allocation, assets)
            return PA
        
def compute_sharpe_ratio_vola_cap(
    data,
    Prices,
    daily_rf_rate,
    previous_weights=None,
    lambda_tc=0.,
):
    
    """
    Compute the portfolio weights that maximize the Sharpe ratio subject to a target volatility cap
    (equal to that of an equal-weight portfolio), optionally penalizing turnover.

    Parameters
    ----------
    data : pd.DataFrame
        Index used to align monthly returns and risk-free rates.
    Prices : pd.DataFrame
        Historical asset prices. Used to compute month-start returns.
    daily_rf_rate : pd.Series or float
        Series (or scalar) of daily risk-free rates. The last available rate is used for excess return.
    previous_weights : array-like, optional
        Weights from the prior period. If provided, a turnover penalty λ_tc·∥w - w_prev∥₁ is applied.
        If None, an equal-weight starting portfolio is assumed and λ_tc is ignored.
    lambda_tc : float, default 0.0
        Turnover penalty coefficient (only active when previous_weights is not None).

    Returns
    -------
    tuple
        sharpe_ann : float
            Annualized Sharpe ratio of the optimized portfolio: (μᵀw - r_f)/sigma(w) x √12 for use as a split score.
        w_opt : np.array
            Vector containing the optimizal weights.

    """

    # 1. Compute monthly returns aligned with data index
    Returns = (Prices
               .resample('MS')
               .ffill()
               .pct_change()
               .shift(-1)
               .dropna())
    Returns = Returns.reindex(data.index).dropna().sort_index()
    assets = Returns.columns
    n = len(assets)

    # 2. Simple historical mean estimate
    mu_vec = Returns.mean().values
    Sigma = Returns.cov()

    # 4. Previous weights for turnover penalty
    w_prev = (previous_weights
              if previous_weights is not None
              else np.ones(n) / n)
    # if no previous weights, no transaction cost
    lambda_tc = lambda_tc if previous_weights is not None else 0.0
    
    # 5. Last available risk-free rate
    rf_val = daily_rf_rate.reindex(Returns.index, method='ffill').iloc[-1]
    rf_rate = float(rf_val) if np.isscalar(rf_val) else float(rf_val.iloc[0])

    # 7. Objective: minimize -Sharpe + gamma * L1 shrinkage + lambda_tc * turnover
    def objective(w):
        w = np.clip(w, 0, None)
        excess = w.dot(mu_vec - rf_rate)
        vol = np.sqrt(w.dot(Sigma.dot(w)))
        sharpe = excess / vol
        turnover = np.sum(np.abs(w - w_prev))
        return -sharpe + lambda_tc * turnover

    # 8. Constraints and bounds: full investment, target variance = equal‐weight variance
    EW = np.ones(n) / n
    target_vol = np.sqrt(EW.dot(Sigma.dot(EW)))
    cons = (
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
        {'type': 'eq', 'fun': lambda w: w.dot(Sigma).dot(w) - target_vol**2}
    )
    bounds = [(0, 1)] * n

    # 9. Run optimizer
    w0 = w_prev
    res = minimize(objective, w0, bounds=bounds, constraints=cons, method='SLSQP')

    # 10. Clean up & normalize
    w_opt = np.clip(res.x, 0, None)
    w_opt /= w_opt.sum()

    # 11. Annualized Sharpe
    excess = w_opt.dot(mu_vec - rf_rate)
    vol = np.sqrt(w_opt.dot(Sigma.dot(w_opt)))
    sharpe_ann = (excess / vol) * np.sqrt(12)

    return sharpe_ann, w_opt

def expanding_window_training_AAF_base_model(MacroEconomic_indicators, Prices, daily_rf_rate):
    """
    Conduct a backtest using an expanding-window evaluation with 6-month out-of-sample periods for training and testing
    the AAF model, from June 2002 through June 2024.

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

            
            Forest = AAF_parallelized(
                Prices,
                ['SP500', 'Bonds', 'Commodities', 'Credits', 'HY'],
                train_macro,
                train_rf,
                compute_sharpe_ratio_vola_cap,
                max_depth=4,
                min_samples_split=72,
                n_trees=360,
                bootstrap_frac=0.9,	
                lambda_tc = 0.,     
                boot = True, 
                previous_weights=weight_last,
                random_state=42
            )

            Forest.fit(features)

            # one allocation call over the full 6-month block
            
            PA_6m = Forest.predict_allocation(test_macro, True)
            _, _, _, weight_6m, _, _ = PA_6m.calculate_PnL_transactions()
            Weights.append(weight_6m)

            last_day = Weights[-1].iloc[-1].name
            weight_last = []

            for key in PA_6m.portfolios[-1].weights.keys():
                weight_last.append(PA_6m.portfolios[-1].weights[key])

        Weights_df = pd.concat(Weights, axis=0)

        Weights_df['Name'] = f"experiment_{j}"

        # collect this run’s list of DataFrames
        Weights_test.append(Weights_df)

    return Weights_test


if __name__ == '__main__':

    start_time = time.time()

    MacroEconomic_indicators, Prices, daily_rf_rate = macro_lagged()

    WT = expanding_window_training_AAF_base_model(MacroEconomic_indicators, Prices, daily_rf_rate)
    

    with open("AAF_TEST_BASE.pkl", "wb") as f:
        pickle.dump(WT, f)

    compare_weights_score_with_baseline(WT, Prices, MacroEconomic_indicators, daily_rf_rate, names = ['AAF_test'])

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Total elapsed time: {elapsed:.2f} seconds")
