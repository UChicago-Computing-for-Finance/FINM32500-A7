import pandas as pd
import polars as pl


def data_ingestion_pandas(file_path: str = 'inputs/market_data-1.csv'):
    """Read CSV into a pandas DataFrame, parse timestamps and set index."""
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.astype('datetime64[us]')
    return df


def data_ingestion_polars(file_path: str = 'inputs/market_data-1.csv'):
    """Read CSV with polars and try to parse dates."""
    df = pl.read_csv(file_path, try_parse_dates=True)
    return df


# Convenience top-level variables used in notebooks/scripts if imported
try:
    df_pandas = data_ingestion_pandas()
except Exception:
    df_pandas = None

try:
    df_polars = data_ingestion_polars()
except Exception:
    df_polars = None
