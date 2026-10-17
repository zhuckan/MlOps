#!/bin/bash
set -e

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?MLFLOW_TRACKING_URI is required}"
MODEL_URI="${MODEL_URI:-models:/creditcard_fraud_model@champion}"
MODEL_PATH="${MODEL_PATH:-/app/model.pkl}"

echo "[entrypoint] MLflow=$MLFLOW_TRACKING_URI model=$MODEL_URI"
python - <<PY
import mlflow, joblib, os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.sklearn.load_model("${MODEL_URI}")
joblib.dump(model, "${MODEL_PATH}")
print(f"[entrypoint] saved {MODEL_PATH}")
PY

exec uvicorn app:app --host 0.0.0.0 --port 8000