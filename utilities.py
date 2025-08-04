import pandas as pd
import numpy as np
import math

class Portfolio:
    """
    Represents a portfolio allocation at a specific date.

    This class encapsulates the asset weights, share quantities, and price data
    for a portfolio initialized with a given investment value.

    Parameters
    ----------
    name : str
        Identifier for this portfolio instance.
    weights : dict
        Mapping from asset name to its target weight in the portfolio (must sum to 1).
    assets : list of str
        Ordered list of asset names included in the portfolio.
    prices : pd.DataFrame
        Historical price series for each asset; index represents dates.
    shares : pd.DataFrame, optional
        Precomputed share quantities for each asset over time. If not provided,
        shares are calculated to match the initial investment at the first date.
    initial_val : float, optional
        Total portfolio value at initialization (default is 1).
    """
    def __init__(self, name, weights, assets, prices, shares = None, initial_val=1):
        self.name = name
        self.init_investment = initial_val
        self.weights = weights
        self.assets = assets
        self.prices = prices

        if shares is None:
            shares_track = {}
            for asset in self.assets:
                shares_track[asset] = self.init_investment * self.weights[asset] / (np.array(self.prices[asset])[0])

            self.shares = pd.DataFrame(shares_track, index = self.prices.index)
        else:
            self.shares = shares

    def add_asset(self, asset_name, asset_price, asset_weight):
        self.assets.append(asset_name)
        if asset_price.index != self.prices.index:
            raise Exception("The prices index don't match")
        self.prices[asset_name] = asset_price
        self.weights[asset_name] = asset_weight

    def Value_Share(self, date):
        val = 0
        for asset in self.assets:
            val += self.shares.loc[date, asset] * self.prices[asset][date]
        return val
    

def geometric_mean_return(returns: pd.Series) -> float:
    """
    Calculate the geometric mean return from a Series of classical returns (as decimals).
    """
    growth_factors = 1 + returns
    product = growth_factors.prod()
    n = len(returns)
    
    # Avoid division by zero or negative roots
    if n == 0:
        return np.nan  
    
    geo_mean = product ** (1 / n) - 1
    return geo_mean

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio for a Series of returns (as decimals).
    
    Sharpe Ratio is defined as:
      (Geometric Mean Return - Risk Free Rate) / Standard Deviation of Returns.
    
    Parameters:
      returns         : pd.Series of periodic returns (as decimals)
      risk_free_rate  : Risk free rate for the period (as decimal), default is 0
      
    Uses ddof=1 for sample standard deviation.
    """
    gm_return = geometric_mean_return(returns)
    excess_return = gm_return - risk_free_rate
    # Compute the sample standard deviation of the returns
    std = returns.std()
    
    # Avoid division by zero if std is zero
    if np.array(std) == 0:
        return np.nan
    
    return excess_return / std, gm_return, std

def calculate_sortino_ratio(excess_returns, target=0):
    """
    Calculate the Sortino ratio.
    
    Parameters:
    - excess_returns: pd.Series of excess returns (returns minus target).
    - returns: pd.Series of total returns.
    - target: The target or minimum acceptable return (default is 0).
    
    Returns:
    - Sortino ratio as a float.
    """
    # Calculate the average excess return
    avg_excess = excess_returns.mean()
    
    # Compute the downside deviation:
    # Select returns that fall below the target.
    downside_returns = excess_returns[excess_returns < target]
    
    # If there are no downside returns, we avoid division by zero
    if downside_returns.empty:
        return np.nan  
    
    # Calculate the downside deviation: sqrt(mean squared deviation below the target)
    downside_deviation = downside_returns.std()
    
    # Compute the Sortino ratio
    sortino_ratio = avg_excess / downside_deviation
    return sortino_ratio 

class Portfolios_Allocation():
    """
    Manage a time-varying asset allocation with a portfolio strategy at each date .

    This class tracks, for each date, which portfolio strategy is in use
    and computes the combined portfolio value based on the historical prices
    of individual assets as well as several performance metrics.

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed price series for each asset in the universe.
    portfolios : list of Portfolio
        A list of `Portfolio` instances representing different allocation strategies.
    allocation : pd.Series or pd.DataFrame
        Date-indexed labels indicating, for each date, which portfolio from `portfolios`
        is active. If a DataFrame, each column can represent a different scenario.
    assets : list of str, optional
        Asset names to include in the aggregation (default is ['SP500', 'Bonds']).
    init_investment : float, optional
        Initial total capital allocated at the start of the backtest (default is 1).
    """
    def __init__(self, prices, portfolios, allocation, assets = ['SP500', 'Bonds'], init_investment = 1):
        self.prices = prices
        self.portfolios = portfolios
        self.allocation = allocation
        self.init_investment = init_investment
        self.assets = assets

        if set(self.assets) < set(self.prices.keys()) :
            raise Exception("We don't have the prices of all assets")
        
        for port in self.portfolios:
            if not math.isclose(port.init_investment, self.init_investment):
                raise Exception("All Portfolios should be initiated with the same initial investment which is the same as the one of a Portfolios Allocation")
        
            if not set(port.assets) <= set(self.assets):
  
                raise Exception("All Portfolios must have a subset of assets of the one specified")
            else:

                if set(port.assets) < set(self.assets):

                    for element in set(self.assets) - set(port.assets):
                        port.add_asset(element, self.prices[element], 0)

    def calculate_PnL_rebalancing(self):
        """
        Compute PnL, returns, cumulative wealth, and turnover for a rebalanced portfolio.

        Calculates profit and loss (excluding transaction costs) for the `Portfolios_Allocation` instance,
        then derives the periodic returns, cumulative wealth path, and portfolio turnover at each rebalance.

        Returns
        -------
            - PnL: profit and loss between rebalancing dates
            - Total Return: PnL / initial_investment
            - Return: DataFrame of monthly returns of the portfolio
            - Weights: Weights used at each date
            - Wealth: cumulative portfolio value over time
            - Turnover: portfolio turnover at each rebalance
        """

        mapping = {}
        for port in self.portfolios:
            mapping[port.name] = [port.weights[asset] for asset in port.assets]
        
        self.allocation.columns = ['Portfolio']
        Explicit_Portfolio = pd.DataFrame({}, index = self.allocation.index)
        Explicit_Portfolio['Portfolio'] = self.allocation['Portfolio'].map(mapping)


        Weights_Description = Explicit_Portfolio["Portfolio"].apply(pd.Series)

        Weights_Description.columns = self.assets

        # Detect weight changes (True where weights differ from the previous row)
        weight_changes = Weights_Description.ne(Weights_Description.shift())

        # DataFrame to hold the number of shares held for each asset
        shares_df = pd.DataFrame(index=self.prices.index, columns=self.assets)


        return_val = pd.DataFrame(index=self.prices.index, columns=["Combined Port Return"])

        turnover_net = pd.DataFrame(index=self.prices.index, columns=["Turnover"])


        shares_df.loc[shares_df.index[0], :] = self.init_investment * Weights_Description.iloc[0] / self.prices.iloc[0]

        port_val = pd.DataFrame(index=self.prices.index, columns=["Portfolio Value"])
        port_val.iloc[0] = self.init_investment
        turnover_net.iloc[0] = 0


        # Loop through all subsequent days
        for t in range(1, len(self.prices)):

            today      = self.prices.index[t]
            yesterday  = self.prices.index[t-1]

            # Value yesterday at yesterday’s prices
            prev_val = port_val.loc[yesterday]

            # Mark‐to‐market at today's prices before trading
            pre_trade_val = (shares_df.loc[yesterday] * self.prices.loc[today]).sum()

            # Actuak weights before rebalancing
            actual_w = (shares_df.loc[yesterday] * self.prices.loc[today]) / pre_trade_val

            # compute realized returns from yesterday → today
            realized = self.prices.loc[today] / self.prices.loc[yesterday] - 1

            # vector‐drifted weights
            drifted_w = Weights_Description.loc[yesterday] * (1 + realized)
            drifted_w /= drifted_w.sum()          

            # Target weights for today
            target_w = Weights_Description.loc[today]

            delta_w  = actual_w - target_w

            turnover_net.loc[today]   = 0.5 * delta_w.abs().sum()

            # Has target changed?
            target_changed = weight_changes.loc[today].any()

            # Has drift exceeded tolerance?
            drift_exceeded = (abs(actual_w - target_w) > 0).any()

            if target_changed or drift_exceeded:
                # Rebalance into target weights
                new_shares = (pre_trade_val * target_w) / self.prices.loc[today]
                shares_df.loc[today] = new_shares

                # Sanity check self‐financing
                check_val = (new_shares * self.prices.loc[today]).sum()
                if not math.isclose(check_val, pre_trade_val, rel_tol=1e-8):
                    raise Exception(f"Not self-financing on {today}! pre={pre_trade_val}, post={check_val}")
            else:
                # No rebalance → carry forward shares
                shares_df.loc[today] = shares_df.loc[yesterday]

            # Record end‐of‐day portfolio value and return
            end_val = (shares_df.loc[today] * self.prices.loc[today]).sum()
            port_val.loc[today]   = end_val
            return_val.loc[today] = ((end_val - prev_val) / prev_val).iloc[0]


        # Select all numerical columns dynamically
        num_columns = shares_df.columns  # Automatically selects all columns

        # Use zip dynamically for n columns
        shares_df['Share Portfolio'] = list(zip(*[shares_df[col] for col in num_columns]))

        Share_Port = shares_df

        last_value = self.allocation['Portfolio'].iloc[-1]

        last_weights = pd.DataFrame(Weights_Description.iloc[-1]).T.iloc[0].to_dict()

        Last_Portfolio = Portfolio(last_value, last_weights, self.assets, self.prices, Share_Port)

        return_diff = Last_Portfolio.Value_Share(Share_Port.index[-1]) - self.init_investment
        tot_return = return_diff/self.init_investment
        
        return return_diff, tot_return, return_val, Weights_Description, port_val, turnover_net
    
    def calculate_Sharpe_Ratio_rebalancing(self, daily_rf_rate, freq = 'M', ret = False, verbose = True, transactions = False, transactions_cost = 0.002):
        """
        Compute performance and risk metrics for a rebalanced portfolio allocation.

        This method calculates periodic excess returns (optionally net of transaction costs)
        and summarizes them into the following metrics:
          - Arithmetic Sharpe Ratio
          - Geometric Sharpe Ratio
          - Sortino Ratio
          - Annualized volatility
          - Maximum drawdown
          - Ulcer Index
          - Turnover

        Parameters
        ----------
        daily_rf_rate : float or pd.Series
            Constant or date-indexed risk-free rate used to compute excess returns.
        freq : str, optional
            Pandas offset alias for return period aggregation (e.g., 'M' for monthly). Defaults to 'M'.
        ret : bool, optional
            If True, return the summary metrics. Defaults to False.
        verbose : bool, optional
            If True, print the computed metrics to the console. Defaults to True.
        transactions : bool, optional
            If True, subtract transaction costs from returns before computing metrics. Defaults to False.
        transactions_cost : float, optional
            Proportional transaction cost rate applied when `transactions=True`. Defaults to 0.002.
        """

        if transactions:
            _ , _ , Returns, WD, port_val, turnover_net = self.calculate_PnL_transactions(transactions_cost)
        else:
            _ , _ , Returns, WD, port_val, turnover_net = self.calculate_PnL_rebalancing()
        Returns = pd.DataFrame(Returns)
        Returns = Returns.dropna()

        total_turnover = turnover_net.sum().iloc[0]

        if freq == 'W':
            RF_Rate  = pd.DataFrame((1 + daily_rf_rate['rf']).resample('W-FRI').prod() - 1)
            RF_Rate = RF_Rate.loc[Returns.index[0]:]
        elif freq == 'M':
            RF_Rate = pd.DataFrame((1 + daily_rf_rate['rf']).resample('MS').prod() - 1)
            RF_Rate = RF_Rate.loc[Returns.index[0]:]
        elif freq == 'D':
            RF_Rate = daily_rf_rate
            RF_Rate = RF_Rate.loc[Returns.index[0]:]


        SR = pd.DataFrame()
        SR['Excess_Return'] = Returns['Combined Port Return'] - RF_Rate['rf']
        Mean_excess = SR.mean()
        Std_excess = SR.std()

        # Calculate the monthly Sharpe ratio
        sharpe_ratio_monthly = Mean_excess / Std_excess

        running_max = port_val.cummax()

        # Calculate the drawdown at each time point
        drawdowns = (port_val - running_max) / running_max

        # Optionally, annualize the Sharpe ratio for weekly or monthly or daily data:
        if freq == 'W':
            sharpe_ratio_annual = sharpe_ratio_monthly * np.sqrt(52)
            geom_sharpe, geom_mean, vola = sharpe_ratio(SR['Excess_Return'])
            sortino = calculate_sortino_ratio(SR['Excess_Return'])
            ulcer_index = np.sqrt((drawdowns.values**2).mean())
            sortino = sortino * np.sqrt(52)
            geom_sharpe_annual = geom_sharpe *np.sqrt(52)

            if verbose:
                print("------------")
                print("Sortino Ratio:", sortino)
                print("------------")
                print("Geometric Mean:", geom_mean)
                print("Weekly Geometric Sharpe Ratio:", geom_sharpe)
                print("Annualized Geometric Sharpe Ratio:", geom_sharpe_annual)
                print("------------")
                print("Arithmetic Mean:", np.array(Mean_excess)[0])
                print("Weekly Arithmetic Sharpe Ratio:", np.array(sharpe_ratio_monthly)[0])
                print("Annualized Arithmetic Sharpe Ratio:", np.array(sharpe_ratio_annual)[0])
                print("------------")
                print("Volatility of Portfolio:", vola)
                print("------------")
                print("Max Drawdown:", np.array(drawdowns.min())[0], "at", drawdowns.idxmin())
                print("Ulcer Index:", ulcer_index)
                print("------------")
                print("Total Turnover:", total_turnover)

            if ret:
                return sortino, geom_sharpe_annual, np.array(sharpe_ratio_annual)[0], vola, np.array(drawdowns.min())[0], ulcer_index, total_turnover

        elif freq == 'M':
            sharpe_ratio_annual = sharpe_ratio_monthly * np.sqrt(12)
            geom_sharpe, geom_mean, vola = sharpe_ratio(SR['Excess_Return'])
            sortino = calculate_sortino_ratio(SR['Excess_Return'])
            sortino = sortino *np.sqrt(12)
            ulcer_index = np.sqrt((drawdowns.values**2).mean())

            if verbose:
                print("------------")
                print("Sortino Ratio:", sortino)
                print("------------")
                print("Geometric Mean:", geom_mean)
                print("Monthly Geometric Sharpe Ratio:", geom_sharpe)
                print("Annualized Geometric Sharpe Ratio:", geom_sharpe * np.sqrt(12))
                print("------------")
                print("Arithmetic Mean:", np.array(Mean_excess)[0])
                print("Weekly Arithmetic Sharpe Ratio:", np.array(sharpe_ratio_monthly)[0])
                print("Annualized Arithmetic Sharpe Ratio:", np.array(sharpe_ratio_annual)[0])
                print("------------")
                print("Volatility of Portfolio:", vola)
                print("------------")
                print("Max Drawdown:", np.array(drawdowns.min())[0], "at", drawdowns.idxmin())
                print("Ulcer Index:", ulcer_index)
                print("------------")
                print("Total Turnover:", total_turnover)

            if ret:
                return sortino, geom_sharpe*np.sqrt(12), np.array(sharpe_ratio_annual)[0], vola, np.array(drawdowns.min())[0], ulcer_index, total_turnover
        
        elif freq == 'D':
            sharpe_ratio_annual = sharpe_ratio_monthly * np.sqrt(252)
            geom_sharpe, geom_mean, vola = sharpe_ratio(SR['Excess_Return'])
            sortino = calculate_sortino_ratio(SR['Excess_Return'])
            sortino = sortino *np.sqrt(252)
            ulcer_index = np.sqrt((drawdowns.values**2).mean())

            if verbose:
                print("------------")
                print("Sortino Ratio:", sortino)
                print("------------")
                print("Geometric Mean:", geom_mean)
                print("Monthly Geometric Sharpe Ratio:", geom_sharpe)
                print("Annualized Geometric Sharpe Ratio:", geom_sharpe * np.sqrt(252))
                print("------------")
                print("Arithmetic Mean:", np.array(Mean_excess)[0])
                print("Weekly Arithmetic Sharpe Ratio:", np.array(sharpe_ratio_monthly)[0])
                print("Annualized Arithmetic Sharpe Ratio:", np.array(sharpe_ratio_annual)[0])
                print("------------")
                print("Volatility of Portfolio:", vola)
                print("------------")
                print("Max Drawdown:", np.array(drawdowns.min())[0], "at", drawdowns.idxmin())
                print("Ulcer Index:", ulcer_index)
                print("------------")
                print("Total Turnover:", total_turnover)


    
    def calculate_PnL_transactions(self, transaction_cost_rate=0.002, print_transaction_costs=False):
        """
        Compute PnL, returns, cumulative wealth, and turnover accounting for transaction costs.

        Calculates profit and loss including transaction costs for each rebalancing period of the `Portfolios_Allocation` instance,
        then derives periodic returns, cumulative portfolio value, and turnover at each rebalance. Optionally prints total transaction costs.

        Parameters
        ----------
        transaction_cost_rate : float, optional
            Proportional cost rate applied to traded volume at each rebalance (default is 0.002).
        print_transaction_costs : bool, optional
            If True, output transaction costs incurred at each rebalance (default is False).

        Returns
        -------
            - PnL: profit and loss between rebalancing dates
            - Total Return: PnL / initial_investment
            - Return: DataFrame of monthly returns of the portfolio
            - Weights: Weights used at each date
            - Wealth: cumulative portfolio value over time
            - Turnover: portfolio turnover at each rebalance
        """

        mapping = {}
        for port in self.portfolios:
            mapping[port.name] = [port.weights[asset] for asset in port.assets]
        
        self.allocation.columns = ['Portfolio']
        Explicit_Portfolio = pd.DataFrame({}, index=self.allocation.index)
        Explicit_Portfolio['Portfolio'] = self.allocation['Portfolio'].map(mapping)

        # Expand the tuple column into separate columns
        Weights_Description = Explicit_Portfolio["Portfolio"].apply(pd.Series)
        Weights_Description.columns = self.assets

        # Detect weight changes
        weight_changes = Weights_Description.ne(Weights_Description.shift())

        # DataFrames for shares, returns, portfolio value, turnover, and now costs
        shares_df            = pd.DataFrame(index=self.prices.index, columns=self.assets)
        return_val           = pd.DataFrame(index=self.prices.index, columns=["Combined Port Return"])
        turnover_net         = pd.DataFrame(index=self.prices.index, columns=["Turnover"])
        transaction_costs    = pd.DataFrame(0.0, index=self.prices.index, columns=["Transaction Cost"])

        # Initialize
        shares_df.iloc[0]    = self.init_investment * Weights_Description.iloc[0] / self.prices.iloc[0]
        port_val             = pd.DataFrame(index=self.prices.index, columns=["Portfolio Value"])
        port_val.iloc[0]     = self.init_investment
        turnover_net.iloc[0] = 0.0

        for t in range(1, len(self.prices)):
            today     = self.prices.index[t]
            yesterday = self.prices.index[t-1]

            prev_val      = port_val.loc[yesterday]["Portfolio Value"]
            pre_trade_val = (shares_df.loc[yesterday] * self.prices.loc[today]).sum()

            # Actual weights before rebalance
            actual_w = (shares_df.loc[yesterday] * self.prices.loc[today]) / pre_trade_val

            # Compute drift and target
            realized  = self.prices.loc[today] / self.prices.loc[yesterday] - 1
            drifted_w = Weights_Description.loc[yesterday] * (1 + realized)
            drifted_w /= drifted_w.sum()
            
            target_w  = Weights_Description.loc[today]
            delta_w   = actual_w - target_w

            turnover = 0.5 * delta_w.abs().sum()
            turnover_net.loc[today] = turnover

            cost = transaction_cost_rate * turnover * pre_trade_val
            transaction_costs.loc[today] = cost

            target_changed  = weight_changes.loc[today].any()
            drift_exceeded  = (abs(actual_w - target_w) > 0).any()

            if target_changed or drift_exceeded:
                net_capital = pre_trade_val - cost
                new_shares  = (net_capital * target_w) / self.prices.loc[today]
                shares_df.loc[today] = new_shares

                # Sanity‐check (ignores cost itself)
                if not math.isclose(new_shares.dot(self.prices.loc[today]), net_capital, rel_tol=1e-8):
                    raise Exception(f"Not self-financing on {today}: pre={net_capital}, post={new_shares.dot(self.prices.loc[today])}")
            else:
                shares_df.loc[today] = shares_df.loc[yesterday]

            raw_end_val = (shares_df.loc[today] * self.prices.loc[today]).sum()
            end_val     = raw_end_val - cost

            port_val.loc[today]   = end_val
            return_val.loc[today] = (end_val - prev_val) / prev_val

        # Optionally print the full cost breakdown and sum
        if print_transaction_costs:
            total_cost = transaction_costs["Transaction Cost"].sum()
            print(f"Total transaction costs paid: {total_cost:.4f}")

        # Final packaging
        num_columns = shares_df.columns
        shares_df['Share Portfolio'] = list(zip(*[shares_df[col] for col in num_columns]))
        Share_Port = shares_df

        last_value   = self.allocation['Portfolio'].iloc[-1]
        last_weights = pd.DataFrame(Weights_Description.iloc[-1]).T.iloc[0].to_dict()
        Last_Portfolio = Portfolio(last_value, last_weights, self.assets, self.prices, Share_Port)

        return_diff = Last_Portfolio.Value_Share(Share_Port.index[-1]) - self.init_investment
        tot_return  = return_diff / self.init_investment

        # return the new costs DataFrame as well
        return return_diff, tot_return, return_val, Weights_Description, port_val, turnover_net 
    
    

