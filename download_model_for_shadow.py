import mlflow
import mlflow.sklearn
import joblib
import os
import sys

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
EXPERIMENT_NAME = "Creditcard_Model_Tuning_v2"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_status = 'Ready_for_staging'",
            order_by=["end_time DESC"],
            max_results=1,
        )
        if not runs_df.empty:
            run_id = runs_df.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            print(f"Found Ready_for_staging model in run {run_id}. Downloading...")
            model = mlflow.sklearn.load_model(model_uri)
            joblib.dump(model, "model.pkl")
            print("Challenger model saved to model.pkl")
            return

    print("WARNING: No Ready_for_staging model found. Falling back to @champion for testing.")
    model_uri = "models:/creditcard_fraud_model@champion"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(model, "model.pkl")
        print("Champion model saved to model.pkl")
    except Exception as e:
        raise SystemExit(f"Failed to download champion model: {e}")


if __name__ == "__main__":
    main()