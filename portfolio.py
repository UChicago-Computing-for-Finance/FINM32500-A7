import pandas as pd
import numpy as np
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from typing import Dict, List, Any

# Load data
market_data = pd.read_csv('inputs/market_data-1.csv', parse_dates=['timestamp'])
portfolio_structure = json.load(open('inputs/portfolio_structure-1.json'))

def compute_position_metrics(symbol: str, quantity: int, market_data, window: int = 20) -> Dict[str, Any]:

    
    symbol_data = market_data[market_data['symbol'] == symbol].copy()
    symbol_data = symbol_data.sort_values('timestamp')
    symbol_data.reset_index(drop=True, inplace=True)
    
    if len(symbol_data) == 0:
        return {
            'symbol': symbol,
            'quantity': quantity,
            'latest_price': None,
            'value': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0
        }
    
    
    latest_price = symbol_data['price'].iloc[-1]
    
    
    value = quantity * latest_price
    
    
    returns = symbol_data['price'].pct_change().dropna()
    
    
    if len(returns) >= window:
        volatility = returns.rolling(window=window).std().iloc[-1]
    else:
        volatility = returns.std() if len(returns) > 1 else 0.0
    
    
    
    running_max = symbol_data['price'].expanding().max()
    
    drawdown = (symbol_data['price'] - running_max) / running_max
    max_drawdown = abs(drawdown.min())  
    
    return {
        'symbol': symbol,
        'quantity': quantity,
        'latest_price': float(latest_price),
        'value': float(value),
        'volatility': float(volatility) if not np.isnan(volatility) else 0.0,
        'max_drawdown': float(max_drawdown)
    }

def aggregate_portfolio_metrics(portfolio: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:

    result = portfolio.copy()
    result['metrics'] = {}
    
    
    position_metrics = []
    for position in portfolio.get('positions', []):
        metrics = compute_position_metrics(
            position['symbol'],
            position['quantity'],
            market_data
        )
        position_metrics.append(metrics)
    
    
    sub_portfolio_metrics = []
    if 'sub_portfolios' in portfolio:
        for sub_portfolio in portfolio['sub_portfolios']:
            aggregated_sub = aggregate_portfolio_metrics(sub_portfolio, market_data)
            sub_portfolio_metrics.append(aggregated_sub)
            result['sub_portfolios'] = [aggregated_sub]
    
    
    all_values = [p['value'] for p in position_metrics]
    all_volatilities = [p['volatility'] for p in position_metrics]
    all_drawdowns = [p['max_drawdown'] for p in position_metrics]
    
    
    for sub in sub_portfolio_metrics:
        if 'metrics' in sub:
            all_values.append(sub['metrics']['total_value'])
            all_drawdowns.append(sub['metrics']['max_drawdown'])
            
    
    
    total_value = sum(all_values)
    
    
    if total_value > 0:
        weights = [p['value'] / total_value for p in position_metrics]
        aggregate_volatility = sum(w * v for w, v in zip(weights, all_volatilities))
    else:
        aggregate_volatility = 0.0
    
    
    
    sub_values = [sub['metrics']['total_value'] for sub in sub_portfolio_metrics]
    sub_volatilities = [sub['metrics']['aggregate_volatility'] for sub in sub_portfolio_metrics]
    total_with_subs = total_value + sum(sub_values)
    
    if total_with_subs > 0:
        
        pos_weights = [p['value'] / total_with_subs for p in position_metrics]
        
        sub_weights = [sv / total_with_subs for sv in sub_values]
        
        aggregate_volatility = (
            sum(w * v for w, v in zip(pos_weights, all_volatilities)) +
            sum(w * v for w, v in zip(sub_weights, sub_volatilities))
        )
    
    
    max_drawdown = max(all_drawdowns) if all_drawdowns else 0.0
    
    
    result['metrics'] = {
        'total_value': float(total_value),
        'aggregate_volatility': float(aggregate_volatility) if not np.isnan(aggregate_volatility) else 0.0,
        'max_drawdown': float(max_drawdown)
    }
    
    
    result['positions'] = position_metrics
    
    return result

def process_position_wrapper(args):
    """Wrapper function for multiprocessing."""
    symbol, quantity, symbol_data_dict, window = args
    
    symbol_df = pd.DataFrame(symbol_data_dict)
    symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
    return compute_position_metrics(symbol, quantity, symbol_df, window)

def aggregate_portfolio_metrics_parallel(portfolio: Dict[str, Any], market_data: pd.DataFrame, market_data_path: str) -> Dict[str, Any]:
    """
    Recursively aggregate metrics using multiprocessing for position computations.
    
    Args:
        portfolio: Portfolio dictionary with positions and sub_portfolios
        market_data: DataFrame with market data (for reference)
        market_data_path: Path to market data CSV (each process will reload it)
    
    Returns:
        Portfolio dictionary with aggregated metrics added
    """
    result = portfolio.copy()
    result['metrics'] = {}
    
    
    
    position_args = []
    for pos in portfolio.get('positions', []):
        symbol = pos['symbol']
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        symbol_data_dict = symbol_data.to_dict('records')
        position_args.append((symbol, pos['quantity'], symbol_data_dict, 20))
    
    
    with Pool(processes=cpu_count()) as pool:
        position_metrics = pool.map(process_position_wrapper, position_args)
    
    
    sub_portfolio_metrics = []
    if 'sub_portfolios' in portfolio:
        result['sub_portfolios'] = []
        for sub_portfolio in portfolio['sub_portfolios']:
            aggregated_sub = aggregate_portfolio_metrics_parallel(sub_portfolio, market_data, market_data_path)
            sub_portfolio_metrics.append(aggregated_sub)
            result['sub_portfolios'].append(aggregated_sub)
    
    
    all_values = [p['value'] for p in position_metrics]
    all_volatilities = [p['volatility'] for p in position_metrics]
    all_drawdowns = [p['max_drawdown'] for p in position_metrics]
    
    for sub in sub_portfolio_metrics:
        if 'metrics' in sub:
            all_values.append(sub['metrics']['total_value'])
            all_drawdowns.append(sub['metrics']['max_drawdown'])
    
    total_value = sum(all_values)
    
    if total_value > 0:
        weights = [p['value'] / total_value for p in position_metrics]
        aggregate_volatility = sum(w * v for w, v in zip(weights, all_volatilities))
    else:
        aggregate_volatility = 0.0
    
    sub_values = [sub['metrics']['total_value'] for sub in sub_portfolio_metrics]
    sub_volatilities = [sub['metrics']['aggregate_volatility'] for sub in sub_portfolio_metrics]
    total_with_subs = total_value + sum(sub_values)
    
    if total_with_subs > 0:
        pos_weights = [p['value'] / total_with_subs for p in position_metrics]
        sub_weights = [sv / total_with_subs for sv in sub_values]
        aggregate_volatility = (
            sum(w * v for w, v in zip(pos_weights, all_volatilities)) +
            sum(w * v for w, v in zip(sub_weights, sub_volatilities))
        )
    
    max_drawdown = max(all_drawdowns) if all_drawdowns else 0.0
    
    result['metrics'] = {
        'total_value': float(total_value),
        'aggregate_volatility': float(aggregate_volatility) if not np.isnan(aggregate_volatility) else 0.0,
        'max_drawdown': float(max_drawdown)
    }
    
    result['positions'] = position_metrics
    
    return result




if __name__ == '__main__':
    # Sequential version
    market_data_path = 'inputs/market_data-1.csv'

    print("Running sequential version...")
    start_sequential = time.time()
    result_sequential = aggregate_portfolio_metrics(portfolio_structure.copy(), market_data)
    end_sequential = time.time()
    sequential_time = end_sequential - start_sequential

    print(f"Sequential time: {sequential_time:.6f} seconds")
    print("\nSequential Result:")
    print(json.dumps(result_sequential, indent=2))

    # Parallel version
    print("\nRunning parallel version...")
    start_parallel = time.time()
    result_parallel = aggregate_portfolio_metrics_parallel(
        portfolio_structure.copy(), 
        market_data, 
        market_data_path
    )
    end_parallel = time.time()
    parallel_time = end_parallel - start_parallel

    print(f"Parallel time: {parallel_time:.6f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    print("\nParallel Result:")
    print(json.dumps(result_parallel, indent=2))

    # Verify results match
    print("\nVerifying results match...")
    result_match = json.dumps(result_sequential, sort_keys=True) == json.dumps(result_parallel, sort_keys=True)
    print(f"Results match: {result_match}")

    # Save output to JSON file
    output_path = 'inputs/portfolio_structure_with_metrics-1.json'
    with open(output_path, 'w') as f:
        json.dump(result_parallel, f, indent=2)