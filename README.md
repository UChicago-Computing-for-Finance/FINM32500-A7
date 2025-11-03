# FINM32500-A7 — Rolling metrics & parallel comparisons

Lightweight project comparing pandas vs polars and threading vs multiprocessing for per-symbol rolling metrics. Includes a notebook, a standalone script, and unit tests.

Contents
- `playground.ipynb` — exploratory notebook (data ingestion, rolling metrics, visualization, threading vs multiprocessing discussion).
- `parrallel.py` — standalone script that runs threaded and process-based experiments and prints timing + resource stats.
- `data_loader.py`, `metrics.py` — helper modules for ingestion and rolling analytics.
- `tests/` — pytest tests validating correctness and consistency.
- `inputs/` — sample input files (CSV and portfolio JSON).

Quick start

1. Install dependencies (recommended in a venv):

```bash
pip install -r requirements.txt
# optional: pip install psutil polars
```

2. Run the notebook
- Open `playground.ipynb` in VS Code / Jupyter and run the cells.

3. Run the script

```bash
python3 parrallel.py
```

4. Run tests

```bash
pytest -q
```

Notes
- `psutil` is optional but recommended for CPU/memory sampling in experiments.
