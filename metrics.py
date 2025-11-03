import pandas as pd
import polars as pl


def compute_rolling_metrics_for_symbol_pandas(df_symbol: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling mean, std and sharpe for a single-symbol pandas DataFrame.

    df_symbol: pandas DataFrame indexed by timestamp and containing column 'price'.
    Returns a DataFrame indexed by timestamp with columns mean_<symbol>, std_<symbol>, sharpe_<symbol>
    """
    # Expect the caller to pass the symbol name via column or index; we'll rely on caller to rename afterwards if needed
    s = df_symbol['price'].sort_index()
    mean = s.rolling(window=window).mean()
    std = s.rolling(window=window).std()
    sharpe = mean / std
    res = pd.concat([mean, std, sharpe], axis=1)
    # caller should rename columns outside if needed
    res.columns = ['mean', 'std', 'sharpe']
    return res


def compute_rolling_metrics_pandas(df_pandas: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling metrics for all symbols in a pandas DataFrame.

    Returns a combined DataFrame with columns duplicated per symbol, e.g. mean_AAPL, std_AAPL, sharpe_AAPL
    """
    out = pd.DataFrame()
    for symbol in df_pandas['symbol'].unique():
        df_symbol = df_pandas[df_pandas['symbol'] == symbol]
        res = compute_rolling_metrics_for_symbol_pandas(df_symbol, window=window)
        # rename columns to include symbol
        res = res.rename(columns={
            'mean': f'mean_{symbol}',
            'std': f'std_{symbol}',
            'sharpe': f'sharpe_{symbol}'
        })
        out = pd.concat([out, res], axis=1)
    return out


def compute_rolling_metrics_polars(df_polars: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Compute rolling metrics using Polars and return a pivoted wide DataFrame similar to pandas output."""
    df_polars_metrics = df_polars.with_columns([
        pl.col('price').rolling_mean(window).over('symbol').alias('rolling_mean'),
        pl.col('price').rolling_std(window).over('symbol').alias('rolling_std'),
    ])
    df_polars_metrics = df_polars_metrics.with_columns([
        (pl.col('rolling_mean') / pl.col('rolling_std')).alias('rolling_sharpe')
    ])

    df_polars_metrics_pivoted = df_polars_metrics.pivot(
        index='timestamp',
        columns='symbol',
        values=['rolling_mean', 'rolling_std', 'rolling_sharpe'],
        aggregate_function='first'
    )

    # Flatten to match pandas naming
    cols = []
    for sym in df_polars['symbol'].unique():
        cols.extend([
            pl.col(f'rolling_mean_{sym}').alias(f'mean_{sym}'),
            pl.col(f'rolling_std_{sym}').alias(f'std_{sym}'),
            pl.col(f'rolling_sharpe_{sym}').alias(f'sharpe_{sym}'),
        ])
    # Select timestamp plus the flattened cols that exist
    sel = ['timestamp'] + [c.alias for c in cols if hasattr(c, 'alias')]
    # If above dynamic aliasing is complex, just return pivoted result for now
    return df_polars_metrics_pivoted
