#!/bin/bash
set -e

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://host.minikube.internal:5001}"
export MODEL_URI="${MODEL_URI:-models:/creditcard_fraud_model@champion}"
export MODEL_PATH="${MODEL_PATH:-/app/model.pkl}"

echo "[entrypoint] MLflow=$MLFLOW_TRACKING_URI model=$MODEL_URI path=$MODEL_PATH"

python -c "
import mlflow
import joblib
import os

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
model = mlflow.sklearn.load_model(os.environ['MODEL_URI'])
joblib.dump(model, os.environ['MODEL_PATH'])
print('Model successfully saved to', os.environ['MODEL_PATH'])
"

exec uvicorn app:app --host 0.0.0.0 --port 8000