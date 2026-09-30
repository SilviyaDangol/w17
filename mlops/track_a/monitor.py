"""Create Evidently feature, target, and custom-threshold drift reports."""
import os
from pathlib import Path

import mlflow
import numpy as np
from evidently import Report
from evidently.metrics import CategoryCount, MeanValue, ValueDrift
from evidently.presets import DataDriftPreset
from evidently.tests import Reference, gt

from .data import load_data

OUT = Path("artifacts/track_a")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_track_a.db"))
    source = load_data()
    reference = source.sample(frac=0.70, random_state=17)
    current = source.drop(reference.index).copy()
    current = _inject_drift(current)
    reports = {
        "data_drift.html": Report([DataDriftPreset()]),
        # ValueDrift on the target is the Evidently-equivalent target-drift check.
        "target_drift.html": Report([ValueDrift(column="Churn")]),
        # Explicit non-default monitored metrics and thresholds are our custom checks.
        "custom_monitoring.html": Report([
            MeanValue(column="MonthlyCharges", tests=[gt(Reference(relative=0.20))]),
            CategoryCount(column="Contract", category="Month-to-month", share_tests=[gt(Reference(relative=0.20))]),
        ]),
    }
    for filename, report in reports.items():
        report.run(current_data=current, reference_data=reference).save_html(str(OUT / filename))
    monthly_charge_shift = float(current.MonthlyCharges.mean() - reference.MonthlyCharges.mean())
    churn_rate_shift = float(current.Churn.mean() - reference.Churn.mean())
    (OUT / "drift_interpretation.md").write_text(
        "# Monitoring interpretation\n\n"
        f"MonthlyCharges mean increased by **{monthly_charge_shift:.2f}** and the target churn rate changed by **{churn_rate_shift:.3f}**. "
        "The current sample was deliberately skewed toward Month-to-month contracts. The Evidently feature, target, and custom-threshold reports flag these engineered changes. "
        "In production, a failed target or custom threshold should create a retraining-review ticket before model promotion.\n"
    )
    mlflow.set_experiment("telco-churn-monitoring")
    with mlflow.start_run(run_name="engineered-drift-check"):
        mlflow.log_metrics({"monthly_charges_mean_shift": monthly_charge_shift, "churn_rate_shift": churn_rate_shift})
        for path in OUT.glob("*.html"):
            mlflow.log_artifact(str(path), artifact_path="evidently_reports")
        mlflow.log_artifact(str(OUT / "drift_interpretation.md"), artifact_path="evidently_reports")


def _inject_drift(current):
    rng = np.random.default_rng(17)
    current["MonthlyCharges"] = current["MonthlyCharges"] + rng.normal(30, 5, len(current))
    month_to_month = rng.random(len(current)) < 0.85
    current.loc[month_to_month, "Contract"] = "Month-to-month"
    current["Churn"] = (rng.random(len(current)) < 0.45).astype(int)
    return current


if __name__ == "__main__":
    main()
