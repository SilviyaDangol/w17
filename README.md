# Week 17 MLOps — Track A: Telco Customer Churn

This branch is deliberately limited to the Week 17 Track A deliverables: reproducible environment, MLflow experimentation and registry, production-model serving, and Evidently monitoring. It does not contain the earlier W15/W16 assistant, UI, Docker files, notebooks, credentials, or prior assignment artifacts.

## Reproduce

```bash
uv sync --extra dev
uv run python -m mlops.track_a.train
uv run python -m mlops.track_a.monitor
uv run pytest
```

`uv.lock` is committed. A clean clone therefore gets the same resolved dependency set with one command. This avoids version drift between scikit-learn, MLflow, Evidently, and their transitive dependencies.

## Dataset

The loader downloads the official 7,043-row [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). It rejects a file with fewer than 7,000 rows so a toy/synthetic replacement cannot silently be used. Downloaded data is intentionally ignored by Git.

## Experiment tracking and selection

The training command runs and compares three materially different models:

| Run | Difference |
|---|---|
| `logistic_c_0_5` | Logistic regression, `C=0.5` |
| `logistic_c_2` | Logistic regression, `C=2.0` |
| `forest_300_depth_8` | Random forest, 300 trees, depth 8 |

Every MLflow run logs hyperparameters; accuracy, precision, recall, F1, and ROC-AUC; a model artifact; a confusion matrix; and an ROC curve. `artifacts/track_a/run_comparison.csv` is the export for side-by-side grading. The winner is selected by F1, then ROC-AUC, rather than accuracy alone because churn is imbalanced. It is registered as `TelcoChurnBest` and transitioned through Staging then Production; `registry_transition.json` preserves this evidence.

The committed run comparison records the actual selected run and its alternatives. For the current official-data run, the random forest won with F1 0.641 and ROC-AUC 0.861; the `C=2` logistic run had F1 0.634 and ROC-AUC 0.857, and the `C=0.5` run had F1 0.632 and ROC-AUC 0.857. The forest is registered because its materially higher recall/F1 is preferred for churn intervention, despite only a modest accuracy gain.

To inspect the local UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow_track_a.db --port 5000
```

## Serving

The API loads the actual Production registry model, not a hard-coded experiment artifact:

```bash
uv run uvicorn mlops.track_a.serve:app --port 8001
```

Send a complete Telco feature row to `POST /predict` as `{"features": {...}}`. Set `MODEL_URI` to use another registry version if needed.

## Monitoring and action

`monitor.py` creates a 70% reference set and 30% current set. It deliberately shifts `MonthlyCharges`, oversamples Month-to-month contracts, and changes the churn rate. It generates and logs three Evidently HTML artifacts:

- `data_drift.html` — feature drift.
- `target_drift.html` — Evidently `ValueDrift` on `Churn`.
- `custom_monitoring.html` — explicit mean-charge and Month-to-month share thresholds using Evidently metric/test APIs.

If target drift or either custom threshold fails in production, the action is a retraining-review ticket, followed by rerunning this comparison and promotion workflow.

Airflow is intentionally not included: it is optional bonus work.
