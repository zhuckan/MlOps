import os
import logging
import joblib
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_PATH = os.getenv("MODEL_PATH", "./model.pkl")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-app")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "dev")
SERVICE_AUTHOR = os.getenv("SERVICE_AUTHOR", "Nikita Zhuck")
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ml-api")

app = FastAPI(title="CreditCard Fraud API")
Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app)

PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Predictions by class",
    ["pred"],
)

logger.info(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
MODEL_LOADED_AT = datetime.now(timezone.utc).isoformat()
logger.info(f"Model loaded: {type(model).__name__}")


class InputData(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float
    V7: float; V8: float; V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float; V17: float; V18: float
    V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float
    Hour: int


class Prediction(BaseModel):
    pred: int
    prob: float


@app.get("/status")
def status():
    return "I'm OK"


@app.get("/version")
def version():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "author": SERVICE_AUTHOR,
        "environment": ENVIRONMENT,
        "model_type": type(model).__name__,
        "model_loaded_at": MODEL_LOADED_AT,
    }


@app.post("/predict", response_model=Prediction)
def predict(form: InputData):
    df = pd.DataFrame.from_dict([form.model_dump()])
    y_proba = float(model.predict_proba(df)[0][1])
    y_pred = int(model.predict(df)[0])
    PREDICTIONS_TOTAL.labels(pred=str(y_pred)).inc()
    return {"pred": y_pred, "prob": y_proba}