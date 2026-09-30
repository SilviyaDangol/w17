"""Train, compare, register, and promote three Telco churn models."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_data

OUT = Path("artifacts/track_a")
MODEL_NAME = "TelcoChurnBest"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_track_a.db"))
    mlflow.set_experiment("telco-churn-training")
    df = load_data().drop(columns=["customerID"])
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    preprocess = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", StandardScaler(), numeric)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=17)
    configurations = [
        ("logistic_c_0_5", LogisticRegression(C=0.5, max_iter=1_000, class_weight="balanced")),
        ("logistic_c_2", LogisticRegression(C=2.0, max_iter=1_000, class_weight="balanced")),
        ("forest_300_depth_8", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=17, class_weight="balanced")),
    ]
    rows = []
    for name, estimator in configurations:
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            predicted = pipeline.predict(X_test)
            probability = pipeline.predict_proba(X_test)[:, 1]
            metrics = _metrics(y_test, predicted, probability)
            mlflow.log_params({"run_name": name, **estimator.get_params()})
            mlflow.log_metrics(metrics)
            figure_path = _plot(y_test, predicted, probability, name)
            mlflow.log_artifact(str(figure_path), artifact_path="evaluation")
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")
            rows.append({"run_id": run.info.run_id, "run_name": name, **metrics})
    comparison = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False)
    comparison.to_csv(OUT / "run_comparison.csv", index=False)
    winner = comparison.iloc[0].to_dict()
    (OUT / "model_selection.json").write_text(json.dumps(winner, indent=2))
    version = _register_and_promote(winner["run_id"])
    (OUT / "registry_transition.json").write_text(json.dumps({"model": MODEL_NAME, "version": version, "transitions": ["Staging", "Production"], "selected_run": winner}, indent=2))
    print(comparison.to_string(index=False))


def _metrics(actual, predicted, probability) -> dict[str, float]:
    return {"accuracy": accuracy_score(actual, predicted), "precision": precision_score(actual, predicted), "recall": recall_score(actual, predicted), "f1": f1_score(actual, predicted), "roc_auc": roc_auc_score(actual, probability)}


def _plot(actual, predicted, probability, name: str) -> Path:
    path = OUT / f"{name}_evaluation.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(actual, predicted, ax=axes[0])
    RocCurveDisplay.from_predictions(actual, probability, ax=axes[1])
    figure.tight_layout(); figure.savefig(path); plt.close(figure)
    return path


def _register_and_promote(run_id: str) -> str:
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    client = MlflowClient()
    client.transition_model_version_stage(MODEL_NAME, version, "Staging", archive_existing_versions=True)
    client.transition_model_version_stage(MODEL_NAME, version, "Production", archive_existing_versions=True)
    return version


if __name__ == "__main__":
    main()
