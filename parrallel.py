#!/usr/bin/env python3
"""Parallel metrics runner

This script runs per-symbol rolling-metric computations using:
 - ThreadPoolExecutor (threads)
 - ProcessPoolExecutor (processes)

It minimizes pickling overhead by writing per-symbol input CSVs which worker
processes read. Each worker writes its output to a CSV; the main process
combines outputs and prints timing / psutil-based resource stats.

Run from the repository root:
    python scripts/parallel_metrics_runner.py

"""
import concurrent.futures
import time
import os
import tempfile
import shutil
from pathlib import Path
import threading
import pandas as pd
import numpy as np
import psutil
import data_loader
import metrics


def compute_metrics_from_csv(symbol: str, input_csv: str, output_csv: str, window: int = 20):
    """Worker function that reads the per-symbol CSV, computes rolling metrics
    using the `metrics` module, and writes the result to output_csv.
    Implemented at module top-level so it's picklable by multiprocessing when run as a script.
    """
    df = pd.read_csv(input_csv, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    # Delegate to metrics.py implementation for pandas per-symbol computation
    res = metrics.compute_rolling_metrics_for_symbol_pandas(df, window=window)
    # rename generic columns to include symbol
    res = res.rename(columns={
        'mean': f'mean_{symbol}',
        'std': f'std_{symbol}',
        'sharpe': f'sharpe_{symbol}'
    })
    res.to_csv(output_csv)
    return output_csv


def _start_monitor(interval=0.05, stop_event=None, records=None):
    if psutil is None:
        return None
    if stop_event is None:
        stop_event = threading.Event()
    if records is None:
        records = {"cpu": [], "mem_mb": [], "ts": []}

    def _run():
        # call cpu_percent once to establish baseline
        psutil.cpu_percent(interval=None)
        while not stop_event.is_set():
            records["cpu"].append(psutil.cpu_percent(interval=None))
            records["mem_mb"].append(psutil.virtual_memory().used / (1024 ** 2))
            records["ts"].append(time.time())
            time.sleep(interval)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return (t, stop_event, records)


def run_workers(worker_cls, tasks, max_workers=None):
    """Run workers where each task is (symbol, input_csv, output_csv).
    Worker function invoked is compute_metrics_from_csv. Returns (duration, records, outputs)
    where outputs is a list of output_csv paths returned by workers.
    """
    records = {"cpu": [], "mem_mb": [], "ts": []}
    monitor = None
    if psutil is not None:
        monitor = _start_monitor(interval=0.05, stop_event=threading.Event(), records=records)

    start = time.time()
    outputs = []
    try:
        with worker_cls(max_workers=max_workers) as ex:
            futures = [ex.submit(compute_metrics_from_csv, sym, in_csv, out_csv)
                       for (sym, in_csv, out_csv) in tasks]
            for f in concurrent.futures.as_completed(futures):
                outputs.append(f.result())
    finally:
        duration = time.time() - start
        if monitor is not None:
            _, stop_event, _ = monitor
            stop_event.set()
            time.sleep(0.05)

    summary = {"duration_s": duration, "cpu_samples": records["cpu"], "mem_samples_mb": records["mem_mb"]}
    return summary, outputs


def combine_outputs(output_paths):
    parts = []
    for p in output_paths:
        df = pd.read_csv(p, parse_dates=["timestamp"]) if "timestamp" in pd.read_csv(p, nrows=0).columns else pd.read_csv(p)
        # The saved CSV from pandas included the index as the first column named 'timestamp'
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        parts.append(df)
    if parts:
        combined = pd.concat(parts, axis=1)
    else:
        combined = pd.DataFrame()
    return combined


def prepare_per_symbol_csvs_from_df(df: pd.DataFrame, tmp_input_dir: str):
    """Prepare per-symbol input/output CSV paths from an in-memory pandas DataFrame.
    Returns list of tasks: (symbol, input_csv, output_csv)
    """
    symbols = df["symbol"].unique().tolist()
    tasks = []
    for sym in symbols:
        sub = df[df["symbol"] == sym].copy()
        out_path = os.path.join(tmp_input_dir, f"input_{sym}.csv")
        sub.to_csv(out_path, index=False)
        tasks.append((sym, out_path, os.path.join(tmp_input_dir, f"output_{sym}.csv")))
    return tasks


def print_summary(name: str, summary: dict):
    print(f"{name} duration: {summary['duration_s']:.4f} s")
    if psutil is not None and len(summary["cpu_samples"]) > 0:
        print(f"{name} mean CPU%: {np.mean(summary['cpu_samples']):.2f}, max CPU%: {np.max(summary['cpu_samples']):.2f}")
        print(f"{name} mean mem (MB): {np.mean(summary['mem_samples_mb']):.2f}, max mem (MB): {np.max(summary['mem_samples_mb']):.2f}")
    else:
        print(f"{name}: psutil not available or no samples collected")


def main():
    # Use data_loader to load data into pandas DataFrame
    df = data_loader.df_pandas
    if df is None:
        # fallback: attempt to read from inputs CSV directly
        repo_root = Path(__file__).resolve().parents[1]
        src_csv = repo_root / "inputs" / "market_data-1.csv"
        if not src_csv.exists():
            print("Input CSV not found at", src_csv)
            return
        df = pd.read_csv(src_csv, parse_dates=["timestamp"]).set_index("timestamp")


    tmpdir = tempfile.mkdtemp(prefix="parallel_metrics_")
    print("Using temp dir:", tmpdir)
    try:
        tasks = prepare_per_symbol_csvs_from_df(df.reset_index(), tmpdir)
        print("Prepared tasks for symbols:", [t[0] for t in tasks])

        # Threading run
        max_threads = min(8, len(tasks))
        print("\n=== ThreadPoolExecutor (threading) ===")
        thread_summary, thread_outputs = run_workers(concurrent.futures.ThreadPoolExecutor, tasks, max_workers=max_threads)
        print_summary("Threading", thread_summary)
        combined_thread = combine_outputs(thread_outputs)
        print("Threading result shape:", combined_thread.shape)

        # Multiprocessing run
        max_procs = min(4, len(tasks))
        print("\n=== ProcessPoolExecutor (multiprocessing) ===")
        proc_summary, proc_outputs = run_workers(concurrent.futures.ProcessPoolExecutor, tasks, max_workers=max_procs)
        print_summary("Multiprocessing", proc_summary)
        combined_proc = combine_outputs(proc_outputs)
        print("Multiprocessing result shape:", combined_proc.shape)

    except Exception as e:
        print("Error during run:", e)
    finally:
        # leave temp dir for inspection if something went wrong, otherwise remove it
        # For safety, only remove if exists and is under system tmp
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
