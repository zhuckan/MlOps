import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from feast import FeatureStore

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")

def main():
    # 1. Инициализация Feast store
    store = FeatureStore(repo_path=FEAST_REPO_PATH)

    # 2. Загрузка entity dataframe (все транзакции из parquet)
    df_raw = pd.read_parquet(os.path.join(FEAST_REPO_PATH, "data", "features.parquet"))
    entity_df = df_raw[["transaction_id", "event_timestamp"]].copy()

    # 3. Получение исторических фич с point-in-time join
    # Убедитесь, что FeatureView называется 'creditcard_features'
    feature_view = store.get_feature_view("creditcard_features")
    feature_refs = [f"{feature_view.name}:{f.name}" for f in feature_view.features]
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs
    ).to_df()

    # 4. Удаляем служебные колонки
    training_df = training_df.drop(columns=["transaction_id", "event_timestamp", "created"], errors="ignore")

    # 5. Разделяем на признаки и целевую переменную
    X = training_df.drop("Class", axis=1)
    y = training_df["Class"]

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Препроцессинг (стандартизация Amount и Hour)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Amount', 'Hour'])
    ], remainder='passthrough')

    # 8. Полный пайплайн (препроцессор + модель)
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    pipeline.fit(X_train, y_train)

    # 9. Оценка качества
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # 10. Логирование в MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Creditcard_Model_Tuning_v2")

    with mlflow.start_run(run_name="Challenger") as run:
        mlflow.log_metrics({"test_roc_auc": roc_auc, "test_auprc": pr_auc})

        # Логируем весь pipeline (важно для последующего использования в API)
        mlflow.sklearn.log_model(pipeline, "model")

        # Сравнение с текущим чемпионом (Production)
        champion_runs = mlflow.search_runs(
            filter_string="tags.model_status = 'Production'",
            order_by=["metrics.test_auprc DESC"]
        )
        if not champion_runs.empty:
            champion_auprc = champion_runs.iloc[0]["metrics.test_auprc"]
            if pr_auc > champion_auprc + 0.01:
                mlflow.set_tag("model_status", "Ready_for_staging")
                print(f"Challenger better: {pr_auc:.4f} > {champion_auprc:.4f}")
            else:
                mlflow.set_tag("model_status", "Rejected")
                print("Challenger not better")
        else:
            mlflow.set_tag("model_status", "Ready_for_staging")
            print("No champion, set Ready_for_staging")

if __name__ == "__main__":
    main()