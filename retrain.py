import json
import os

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")
HOLDOUT_FRACTION = 0.2
EXPERIMENT_NAME = "Creditcard_Model_Tuning_v2"
HOLDOUT_VERSION = "frozen_holdout_v1"
HOLDOUT_TAG_KEY = "frozen_holdout_version"


def load_frozen_holdout():
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("[holdout] Experiment not found → no frozen holdout yet")
        return None, None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.{HOLDOUT_TAG_KEY} = '{HOLDOUT_VERSION}'",
        max_results=1,
    )
    if not runs:
        print(f"[holdout] No run with tag {HOLDOUT_TAG_KEY}={HOLDOUT_VERSION}")
        return None, None

    holdout_run_id = runs[0].info.run_id
    print(f"[holdout] Found frozen holdout in run {holdout_run_id}")

    holdout_path = mlflow.artifacts.download_artifacts(
        run_id=holdout_run_id, artifact_path="holdout.parquet"
    )
    meta_path = mlflow.artifacts.download_artifacts(
        run_id=holdout_run_id, artifact_path="holdout_meta.json"
    )

    holdout_df = pd.read_parquet(holdout_path)
    with open(meta_path) as f:
        meta = json.load(f)
    cutoff = pd.Timestamp(meta["cutoff"])

    print(f"[holdout] Loaded: {len(holdout_df)} rows, cutoff={cutoff}")
    return holdout_df, cutoff


def save_frozen_holdout(holdout_df, cutoff, parent_run_id):
    client = mlflow.tracking.MlflowClient()
    holdout_path = "/tmp/holdout.parquet"
    meta_path = "/tmp/holdout_meta.json"

    holdout_df.to_parquet(holdout_path, index=False)
    meta = {
        "cutoff": str(cutoff),
        "created_at": str(pd.Timestamp.utcnow()),
        "rows": len(holdout_df),
        "version": HOLDOUT_VERSION,
        "holdout_fraction": HOLDOUT_FRACTION,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    client.log_artifact(parent_run_id, holdout_path, artifact_path="")
    client.log_artifact(parent_run_id, meta_path, artifact_path="")
    client.set_tag(parent_run_id, HOLDOUT_TAG_KEY, HOLDOUT_VERSION)
    print(f"[holdout] Saved to MLflow run {parent_run_id}: {len(holdout_df)} rows, cutoff={cutoff}")


def load_features_and_targets(store):
    df_features_raw = pd.read_parquet(
        os.path.join(FEAST_REPO_PATH, "data", "features.parquet")
    )
    entity_df_features = df_features_raw[["transaction_id", "event_timestamp"]].copy()

    fv_features = store.get_feature_view("creditcard_features")
    features_refs = [f"{fv_features.name}:{f.name}" for f in fv_features.features]
    features_df = store.get_historical_features(
        entity_df=entity_df_features,
        features=features_refs,
    ).to_df()

    df_targets_raw = pd.read_parquet(
        os.path.join(FEAST_REPO_PATH, "data", "targets.parquet")
    )
    entity_df_targets = df_targets_raw[["transaction_id", "target_timestamp"]].copy()

    fv_targets = store.get_feature_view("creditcard_targets")
    targets_refs = [f"{fv_targets.name}:{f.name}" for f in fv_targets.features]
    targets_df = store.get_historical_features(
        entity_df=entity_df_targets,
        features=targets_refs,
    ).to_df()

    return features_df.merge(
        targets_df[["transaction_id", "Class"]],
        on="transaction_id",
        how="inner",
    )


def build_pipeline():
    preprocessor = ColumnTransformer(
        [('num', StandardScaler(), ['Amount', 'Hour'])],
        remainder='passthrough',
    )
    return Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(random_state=42, eval_metric='logloss')),
    ])


def temporal_split(df, holdout_fraction=0.2):
    if len(df) < 2:
        raise ValueError("Not enough data to split")
    if df["Class"].nunique() < 2:
        raise ValueError("Only one class present in dataset")
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - holdout_fraction))
    train_df = df.iloc[:split_idx].copy()
    holdout_df = df.iloc[split_idx:].copy()
    return train_df, holdout_df


def main():
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    df = load_features_and_targets(store)
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    print(f"[data] Total rows: {len(df)}, "
          f"{df['event_timestamp'].min()} → {df['event_timestamp'].max()}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    holdout_df, cutoff = load_frozen_holdout()

    with mlflow.start_run(run_name="Challenger") as run:
        run_id = run.info.run_id

        if holdout_df is None:
            split_idx = int(len(df) * (1 - HOLDOUT_FRACTION))
            cutoff = df.iloc[split_idx]["event_timestamp"]
            holdout_df = df.iloc[split_idx:].copy()
            train_df = df.iloc[:split_idx].copy()
            save_frozen_holdout(holdout_df, cutoff, run_id)
        else:
            train_df = df[df["event_timestamp"] < cutoff].copy()

        if len(train_df) == 0 or len(holdout_df) == 0:
            raise ValueError(
                f"Empty split: train={len(train_df)}, holdout={len(holdout_df)}"
            )
        if train_df["Class"].nunique() < 2:
            raise ValueError(f"Train has only one class: {train_df['Class'].unique()}")

        print(f"[split] Train:   {len(train_df)} rows "
              f"({train_df['event_timestamp'].min()} → {train_df['event_timestamp'].max()})")
        print(f"[split] Holdout: {len(holdout_df)} rows "
              f"({holdout_df['event_timestamp'].min()} → {holdout_df['event_timestamp'].max()})")

        feature_cols = [
            c for c in df.columns
            if c not in ["transaction_id", "event_timestamp", "created", "Class"]
        ]
        X_train = train_df[feature_cols]
        y_train = train_df["Class"]
        X_holdout = holdout_df[feature_cols]
        y_holdout = holdout_df["Class"]

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_holdout)[:, 1]
        roc_auc = roc_auc_score(y_holdout, y_pred_proba)
        pr_auc = average_precision_score(y_holdout, y_pred_proba)

        mlflow.log_param("holdout_fraction", HOLDOUT_FRACTION)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("holdout_rows", len(holdout_df))
        mlflow.log_param("holdout_cutoff", str(cutoff))
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_auprc", pr_auc)
        mlflow.set_tag("dataset_version", HOLDOUT_VERSION)
        mlflow.set_tag("holdout_cutoff", str(cutoff))
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name="creditcard_fraud_model",
        )

        champion_runs = mlflow.search_runs(
            experiment_ids=[run.info.experiment_id],
            filter_string="tags.model_status = 'Production'",
            order_by=["metrics.test_auprc DESC"],
        )

        if champion_runs.empty:
            mlflow.set_tag("model_status", "Ready_for_staging")
            print("[compare] No champion → Ready_for_staging")
        else:
            champion_run_id = champion_runs.iloc[0]["run_id"]
            print(f"[compare] Champion run_id = {champion_run_id}")
            champion_model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
            champ_pred_proba = champion_model.predict_proba(X_holdout)[:, 1]
            champion_auprc = average_precision_score(y_holdout, champ_pred_proba)
            print(f"[compare] Challenger AUPRC = {pr_auc:.4f}, "
                  f"Champion AUPRC (on same holdout) = {champion_auprc:.4f}")
            mlflow.log_metric("champion_auprc_on_same_holdout", champion_auprc)
            if pr_auc > champion_auprc + 0.01:
                mlflow.set_tag("model_status", "Ready_for_staging")
            else:
                mlflow.set_tag("model_status", "Rejected")


if __name__ == "__main__":
    main()