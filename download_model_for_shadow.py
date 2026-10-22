import os

import joblib
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
EXPERIMENT_NAME = "Creditcard_Model_Tuning_v2"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"Experiment {EXPERIMENT_NAME} not found")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_status = 'Ready_for_staging'",
        order_by=["end_time DESC"]
    )
    if runs_df.empty:
        raise SystemExit("No Ready_for_staging model found")

    run_id = runs_df.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    joblib.dump(model, "model.pkl")
    print(f"Model from run {run_id} saved to model.pkl")

if __name__ == "__main__":
    main()