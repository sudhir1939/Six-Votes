# Six Votes: Model Evaluation and Knowledge Distillation

Six Votes is a NumPy-based machine-learning assessment project covering model comparison, panel evaluation, and knowledge distillation. The project evaluates prediction models, analyzes ensemble behavior, trains a digit-classification teacher, and tests whether a student can inherit the teacher's representations through auxiliary outputs.

## Key Features

- Compares model accuracy using bootstrap confidence intervals
- Performs McNemar statistical testing for paired model predictions
- Measures panel diversity with phi correlation and effective panel size
- Evaluates plurality and majority voting strategies
- Trains and compares TinyNet teacher and student networks
- Generates machine-readable JSON results and a Markdown report
- Includes public tests and a submission validation script

## Project Parts

| Module | Purpose |
|---|---|
| `sixvotes/leaderboard.py` | Accuracy, ranking, bootstrap intervals, and significance tests |
| `sixvotes/panel.py` | Model disagreement, effective panel size, and voting accuracy |
| `sixvotes/distill.py` | Teacher training and student distillation experiments |
| `sixvotes/tinynet.py` | Lightweight NumPy neural network implementation |
| `sixvotes/report.py` | Markdown report generation |
| `sixvotes/io.py` | Dataset loading and CSV/JSON output helpers |

## Setup

Python 3.10 or newer is required. NumPy and pytest are the project dependencies.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the Analysis

From the project root:

```powershell
python -m sixvotes audit --predictions data/panel_predictions.csv --out outputs
python -m sixvotes distill --digits data/digits.npz --out outputs
python check_submission.py
```

The commands generate:

- `outputs/results.json` - leaderboard and panel analysis
- `outputs/REPORT.md` - human-readable findings
- `outputs/subliminal_results.json` - distillation experiment results

## Run Tests

```powershell
pytest -q
```

## Repository Guide

| Path | Description |
|---|---|
| `sixvotes/` | Core implementation |
| `data/` | Digit and panel-prediction datasets |
| `tests/` | Public unit tests |
| `docs/` | Output contract and supporting reading |
| `submission/` | Notes, findings, and one-pager deliverables |
| `outputs/` | Generated audit and distillation results |
| `Six_Votes_Starter.ipynb` | Guided notebook version of the assessment |
| `ASSIGNMENT_BRIEF.pdf` | Original project brief |
