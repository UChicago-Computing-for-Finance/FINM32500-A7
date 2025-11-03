# Performance Report: Pandas vs Polars

## Performance Comparison Table

| Metric | Pandas | Polars | Speedup Ratio | Winner |
|--------|--------|--------|---------------|--------|
| **Data Ingestion Time** | 0.098 seconds | 0.005 seconds | **20.5x** | Polars |
| **Rolling Metrics Computation** | 1.168 seconds | 0.032 seconds | **36.8x** | Polars |
| **Memory Usage** | 19.65 MB | 19.65 MB | 1.0x | Equal* |
| **Parallel Execution Speed** | Sequential: 0.183s<br>Multiprocessing: ~0.1-0.15s | N/A** | 1.2-1.8x | Multiprocessing |

---

## Analysis of Performance Metrics

### Data Ingestion Time

**Pandas (0.098s):** Uses Python-based CSV parser with row-by-row processing. The `parse_dates` operation adds overhead as timestamps are parsed sequentially.

**Polars (0.005s):** Leverages Rust-based CSV parsing engine with native parallelism. 

---

### Rolling Metrics Computation

**Pandas (1.168s):** 
- Requires explicit iteration through each symbol
- Uses sequential `DataFrame.rolling()` operations per symbol
- Multiple `pd.concat()` operations to combine results incrementally
- Row-oriented processing model with Python overhead

**Polars (0.032s):**
- Uses `.over("symbol")` for grouped rolling calculations without loops
- Vectorized operations across all symbols simultaneously
- Single-pass computation leveraging columnar storage

---

### Memory Usage

**Both: 19.65 MB** (measured after Polars-to-Pandas conversion)

**Pandas:** Row-oriented storage with Python object overhead. Each row stored as a Python object with associated metadata.

**Polars:** Columnar storage using Apache Arrow format. Better memory efficiency through:
- Data type optimization (smaller numeric types where appropriate)
- Compression for repeated values
- Zero-copy operations possible
- No Python object overhead

---

### Parallel Execution Speed

**Sequential (0.183s):** Processes positions one at a time, no parallelization overhead.

**Multiprocessing (~0.1-0.15s):** 
- Distributes position computations across multiple CPU cores
- Overhead from process creation, data serialization (pickling), and result aggregation
- Benefits increase with more positions and more complex per-position computations