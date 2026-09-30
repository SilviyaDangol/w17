"""FastAPI serving endpoint that loads the Production registry model."""
import os
from functools import lru_cache
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Telco Churn Production Model")
MODEL_URI = os.getenv("MODEL_URI", "models:/TelcoChurnBest/Production")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(description="One complete Telco feature row, excluding customerID and Churn.")


@lru_cache
def get_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_track_a.db"))
    return mlflow.sklearn.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | bool]:
    probability = float(get_model().predict_proba(pd.DataFrame([request.features]))[0, 1])
    return {"churn_probability": probability, "churn": probability >= 0.5}
