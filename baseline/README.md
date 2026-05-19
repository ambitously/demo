# RocoBench Evaluation Tools

Batch evaluation and code packing tools for the RocoBench project.

## Environment

This repository is managed with `uv` and pins Python 3.8 in
`.python-version`.

```bash
uv sync
uv run python evaluator.py
```

The first `uv sync` will create `.venv` and, if needed, download a compatible
Python 3.8 interpreter.

Each evaluator run writes its terminal log, task logs, and run artifacts into a
single per-run directory:

```text
output/run_YYYYMMDD_HHMMSS/
├── evaluator.log
├── run.json
├── summary.json
└── tasks/
    └── 01_sort/
        ├── command.txt
        ├── runs/
        │   ├── args_YYYYMM_HHMM.json
        │   └── run_0/
        ├── stdout.log
        ├── stderr.log
        ├── result.json
        └── summary.json
```

## evaluator.py - Batch Task Evaluation

Automated batch testing tool for multiple robot tasks with timeout control and colored output.

> **⚠️ Important Note:** During grading/testing, `evaluator.py` will be replaced with the original code. Any modifications to this file will not affect the final evaluation results.

### Usage

```bash
uv run python evaluator.py
```

### Configuration

Edit the script to configure tasks:

```python
# Configure per-task timeouts (seconds)
DEFAULT_RUN_TIMEOUTS = {
    "sort": 600,
    "cabinet": 600,
    "rope": 600,
    "sweep": 600,
    "sandwich": 600,
    "pack": 600,
}

# Select tasks to run
results.append(test_run_dialog("sort", 5, "output"))
results.append(test_run_dialog("cabinet", 5, "output"))
# Uncomment tasks as needed
```

## pack_code.sh - Workspace Packing

Creates a zip archive of the workspace while respecting `.gitignore` rules.

### Usage

```bash
# Use default filename (code_YYYYMMDD_HHMMSS.zip)
./pack_code.sh

# Specify custom filename
./pack_code.sh myproject.zip

# Auto-adds .zip extension if missing
./pack_code.sh myproject
```

## Notes

- Script requires executable permission: `chmod +x pack_code.sh`
