import os
import tempfile
import pandas as pd
import numpy as np
import pytest

import sys
from pathlib import Path

# Ensure repo root is on sys.path so local modules (metrics, parrallel) can be imported in tests
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import metrics
import parrallel
import concurrent.futures as cf


def test_compute_rolling_metrics_for_symbol_pandas():
    # simple deterministic series
    ts = pd.date_range("2020-01-01", periods=5, freq="T")
    df = pd.DataFrame({"timestamp": ts, "price": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df = df.set_index("timestamp")

    res = metrics.compute_rolling_metrics_for_symbol_pandas(df, window=3)

    expected_mean = df["price"].rolling(3).mean()
    expected_std = df["price"].rolling(3).std()
    expected_sharpe = expected_mean / expected_std

    pd.testing.assert_series_equal(res["mean"], expected_mean, check_names=False)
    pd.testing.assert_series_equal(res["std"], expected_std, check_names=False)
    # allow NaNs and floating tolerance
    np.testing.assert_allclose(res["sharpe"].fillna(0), expected_sharpe.fillna(0), rtol=1e-6, atol=1e-9)


def test_threading_vs_multiprocessing_consistency():
    # create small dataset with two symbols
    ts = pd.date_range("2020-01-01", periods=20, freq="T")
    rows = []
    for sym in ["A", "B"]:
        for i, t in enumerate(ts):
            rows.append({"timestamp": t, "symbol": sym, "price": float(i + (0 if sym == "A" else 0.5))})
    df = pd.DataFrame(rows)

    tmpdir = tempfile.mkdtemp(prefix="test_parallel_")
    try:
        tasks = parrallel.prepare_per_symbol_csvs_from_df(df, tmpdir)

        t_summary, t_outputs = parrallel.run_workers(cf.ThreadPoolExecutor, tasks, max_workers=2)
        p_summary, p_outputs = parrallel.run_workers(cf.ProcessPoolExecutor, tasks, max_workers=2)

        t_comb = parrallel.combine_outputs(t_outputs)
        p_comb = parrallel.combine_outputs(p_outputs)

        # align columns and index
        t_comb = t_comb.sort_index().sort_index(axis=1)
        p_comb = p_comb.sort_index().sort_index(axis=1)

        # Values should be equal (floating tolerance)
        assert t_comb.shape == p_comb.shape
        np.testing.assert_allclose(t_comb.fillna(0).values, p_comb.fillna(0).values, rtol=1e-6, atol=1e-9)
    finally:
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except Exception:
            pass


def test_pandas_vs_polars_single_symbol_equivalence():
    try:
        import polars as pl
    except Exception:
        pytest.skip("polars not installed")

    ts = pd.date_range("2020-01-01", periods=10, freq="T")
    df = pd.DataFrame({"timestamp": ts, "symbol": ["X"] * len(ts), "price": np.arange(len(ts), dtype=float)})

    # pandas result
    panda_res = metrics.compute_rolling_metrics_pandas(df, window=3)

    # polars input
    pl_df = pl.from_pandas(df)
    pol_res = metrics.compute_rolling_metrics_polars(pl_df, window=3)

    # convert polars pivot result to pandas and rename expected columns
    pol_pd = pol_res.to_pandas()
    # polars pivot uses rolling_mean_X etc. map to mean_X
    cols_map = {c: c.replace('rolling_mean_', 'mean_').replace('rolling_std_', 'std_').replace('rolling_sharpe_', 'sharpe_') for c in pol_pd.columns}
    pol_pd = pol_pd.rename(columns=cols_map)

    # extract columns for symbol X
    # pandas result columns are like mean_X, std_X, sharpe_X
    # Depending on implementation, align by selecting overlapping columns
    common = set(panda_res.columns).intersection(pol_pd.columns)
    assert len(common) > 0
    panda_sel = panda_res.loc[:, sorted(common)].sort_index()
    pol_sel = pol_pd.loc[:, sorted(common)].sort_index()

    np.testing.assert_allclose(panda_sel.fillna(0).values, pol_sel.fillna(0).values, rtol=1e-6, atol=1e-9)


def test_portfolio_aggregation_totals():
    # Build a tiny market data set and portfolio
    ts = pd.to_datetime(["2025-01-01T10:00:00", "2025-01-01T10:01:00"]) 
    rows = [
        {"timestamp": ts[0], "symbol": "AAA", "price": 10.0},
        {"timestamp": ts[1], "symbol": "AAA", "price": 12.0},
        {"timestamp": ts[0], "symbol": "BBB", "price": 5.0},
        {"timestamp": ts[1], "symbol": "BBB", "price": 6.0},
    ]
    market_df = pd.DataFrame(rows)

    portfolio = {
        "name": "test",
        "positions": [
            {"symbol": "AAA", "quantity": 2},
            {"symbol": "BBB", "quantity": 3},
        ]
    }

    # Sequential aggregation
    seq = __import__('portfolio').aggregate_portfolio_metrics(portfolio.copy(), market_df)

    # Expected total_value = latest_price * qty summed
    latest_prices = {sym: market_df[market_df['symbol'] == sym].sort_values('timestamp')['price'].iloc[-1]
                     for sym in ['AAA', 'BBB']}
    expected_total = latest_prices['AAA'] * 2 + latest_prices['BBB'] * 3

    assert pytest.approx(expected_total, rel=1e-9) == seq['metrics']['total_value']

    # Parallel aggregation should match sequential for these inputs
    par = __import__('portfolio').aggregate_portfolio_metrics_parallel(portfolio.copy(), market_df, market_data_path='')
    assert pytest.approx(seq['metrics']['total_value'], rel=1e-9) == par['metrics']['total_value']
