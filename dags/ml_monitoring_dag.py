import json
import os
from datetime import datetime, timedelta
from io import BytesIO

import mlflow
import pandas as pd
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.python import PythonSensor
from kubernetes.client import models as k8s
from minio import Minio

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

GITLAB_URL = os.getenv("GITLAB_URL", "http://host.docker.internal:8088")
GITLAB_PROJECT_ID = os.getenv("GITLAB_PROJECT_ID", "1")
GITLAB_TRIGGER_TOKEN = os.getenv("GITLAB_TRIGGER_TOKEN", "")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI_AIRFLOW", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Creditcard_Model_Tuning_v2")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "host.docker.internal:9050")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "traffic-logs")
BASELINE_POINTER = os.getenv("BASELINE_POINTER", "baseline_current.txt")

FEATURES_PARQUET = os.getenv(
    "FEATURES_PARQUET",
    "/opt/airflow/feature_repo/data/features.parquet",
)

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/opt/airflow/feature_repo")
CURRENT_WINDOW_HOURS = os.getenv("CURRENT_WINDOW_HOURS", "168")
RETRAIN_JOB_VERSION = os.getenv("RETRAIN_JOB_VERSION", "v7")


def _send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[telegram] Credentials not set. Message: {message}")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        print(f"[telegram] {r.status_code}")
    except Exception as e:
        print(f"[telegram] Failed: {e}")


def _get_drift_result(context):
    result = context['ti'].xcom_pull(task_ids='drift_check') or {}
    if not isinstance(result, dict):
        return {"drift_detected": False, "report_url": ""}
    return result


def send_alert(**context):
    result = _get_drift_result(context)
    if not result.get("drift_detected"):
        return
    report_url = result.get("report_url", "")
    _send_telegram(f"Data Drift detected! Report: {report_url}")


def should_continue(**context):
    result = _get_drift_result(context)
    return 'send_alert' if result.get("drift_detected") else 'no_drift'


def no_drift():
    print("No drift detected, skipping retraining.")


def notify_ready(**context):
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("Experiment not found")
        return

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_status = 'Ready_for_staging'",
        order_by=["end_time DESC"],
        max_results=1,
    )
    if runs_df.empty:
        print("No Ready_for_staging model found.")
        return

    run_id = runs_df.iloc[0]["run_id"]
    auprc = runs_df.iloc[0].get("metrics.test_auprc", "N/A")
    _send_telegram(
        f"Новая модель готова к апруву\n"
        f"Run ID: {run_id}\n"
        f"AUPRC: {auprc}\n\n"
        f"ОДОБРИТЬ: Airflow -> Admin -> Variables\n"
        f"  Key: ml_approval_{run_id}\n"
        f"  Value: approve\n\n"
        f"ОТКЛОНИТЬ: Value = reject\n"
        f"Таймаут ожидания: 24 часа"
    )


def approval_sensor(**context):
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("[approval] Experiment not found, waiting...")
        return False

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.model_status = 'Ready_for_staging'",
        order_by=["end_time DESC"],
        max_results=1,
    )
    if runs.empty:
        print("[approval] No Ready_for_staging model, waiting...")
        return False

    run_id = runs.iloc[0]["run_id"]
    var_key = f"ml_approval_{run_id}"
    approval = Variable.get(var_key, default_var=None)

    if approval == "approve":
        print(f"[approval] run {run_id} approved")
        context['ti'].xcom_push(key='approved_run_id', value=run_id)
        return True
    if approval == "reject":
        raise Exception(f"[approval] run {run_id} rejected by operator")

    print(f"[approval] Waiting for variable {var_key}")
    return False


def promote_to_production(**context):
    run_id = context['ti'].xcom_pull(task_ids='wait_for_approval', key='approved_run_id')
    if not run_id:
        raise Exception("[promote] approved_run_id not found in XCom")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "model_status", "Production")
    client.set_tag(run_id, "approved_at", str(pd.Timestamp.utcnow()))
    Variable.delete(f"ml_approval_{run_id}")

    context['ti'].xcom_push(key='promoted_run_id', value=run_id)
    print(f"[promote] run {run_id} -> Production")


def refresh_baseline(**context):
    run_id = context['ti'].xcom_pull(task_ids='promote_to_production', key='promoted_run_id')
    if not run_id:
        raise Exception("[refresh_baseline] promoted_run_id not found")

    features_df = pd.read_parquet(FEATURES_PARQUET)
    feature_cols = [
        c for c in features_df.columns
        if c not in ['transaction_id', 'event_timestamp', 'created', 'Class']
    ]
    baseline_df = features_df[feature_cols].copy()

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)

    tag = f"promoted_{run_id[:8]}"
    object_name = f"baseline_{tag}.csv"
    meta_name = f"baseline_{tag}.meta.json"

    csv_bytes = baseline_df.to_csv(index=False).encode("utf-8")
    client.put_object(MINIO_BUCKET, object_name, BytesIO(csv_bytes), len(csv_bytes))

    meta = {
        "version": f"baseline_{tag}",
        "created_at": str(pd.Timestamp.utcnow()),
        "rows": len(baseline_df),
        "mlflow_run_id": run_id,
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    client.put_object(MINIO_BUCKET, meta_name, BytesIO(meta_bytes), len(meta_bytes))

    pointer_bytes = object_name.encode("utf-8")
    client.put_object(
        MINIO_BUCKET, BASELINE_POINTER, BytesIO(pointer_bytes), len(pointer_bytes)
    )

    print(f"[refresh_baseline] {object_name} ({len(baseline_df)} rows), run_id={run_id}")


def notify_approved(**context):
    run_id = context['ti'].xcom_pull(task_ids='promote_to_production', key='promoted_run_id')
    _send_telegram(
        f"Модель ОДОБРЕНА\n"
        f"Run ID: {run_id}\n"
        f"Статус: Production\n"
        f"Запускаю деплой через GitLab..."
    )


def trigger_gitlab_pipeline(**context):
    if not GITLAB_TRIGGER_TOKEN:
        print("GITLAB_TRIGGER_TOKEN not set. Skipping trigger.")
        return
    url = f"{GITLAB_URL}/api/v4/projects/{GITLAB_PROJECT_ID}/trigger/pipeline"
    payload = {
        "token": GITLAB_TRIGGER_TOKEN,
        "ref": "main",
        "variables[PROD_DEPLOY]": "true",
    }
    try:
        r = requests.post(url, data=payload, timeout=30)
        print(f"GitLab trigger response: {r.status_code}, {r.text[:300]}")
    except Exception as e:
        print(f"Failed to trigger GitLab pipeline: {e}")


with DAG(
    dag_id='ml_monitoring_continuous_training',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
    tags=['ml', 'monitoring'],
) as dag:

    drift_check = KubernetesPodOperator(
        task_id='drift_check',
        name='drift-monitor',
        namespace='ml-service',
        image='drift-monitor:v2',
        image_pull_policy='Never',
        cmds=['python', 'drift_monitor.py'],
        env_vars={
            'MINIO_ENDPOINT': 'minio.ml-service.svc.cluster.local:9000',
            'MINIO_ACCESS_KEY': MINIO_ACCESS_KEY,
            'MINIO_SECRET_KEY': MINIO_SECRET_KEY,
            'PUSHGATEWAY_URL': 'http://pushgateway.monitoring.svc.cluster.local:9091',
            'FEAST_REPO_PATH': '/app/feature_repo',
            'CURRENT_WINDOW_HOURS': str(CURRENT_WINDOW_HOURS),
        },
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "512Mi", "cpu": "250m"},
            limits={"memory": "2Gi", "cpu": "1"},
        ),
        do_xcom_push=True,
        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=False,
        config_file="/opt/airflow/kubeconfig-ci.yaml",
    )

    branch = BranchPythonOperator(
        task_id='branch_drift',
        python_callable=should_continue,
        provide_context=True,
    )

    no_drift_task = PythonOperator(
        task_id='no_drift',
        python_callable=no_drift,
    )

    alert = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert,
        provide_context=True,
    )

    retrain_model = KubernetesPodOperator(
        task_id='retrain_model',
        namespace='ml-service',
        image=f'retrain-job:{RETRAIN_JOB_VERSION}',
        image_pull_policy='Never',
        name='retrain-job',
        cmds=["python", "retrain.py"],
        env_vars={
            'MLFLOW_TRACKING_URI': 'http://mlflow.ml-service.svc.cluster.local:5001',
            'FEAST_REPO_PATH': '/app/feature_repo',
            'MLFLOW_ALLOW_FILE_STORE': 'true',
        },
        container_resources=k8s.V1ResourceRequirements(
            requests={"memory": "1Gi", "cpu": "500m"},
            limits={"memory": "3Gi", "cpu": "2"},
        ),
        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=False,
        config_file="/opt/airflow/kubeconfig-ci.yaml",
    )

    notify_ready_task = PythonOperator(
        task_id='notify_ready',
        python_callable=notify_ready,
        provide_context=True,
    )

    wait_approval = PythonSensor(
        task_id='wait_for_approval',
        python_callable=approval_sensor,
        poke_interval=120,
        timeout=3600 * 24,
        mode='reschedule',
    )

    promote_model = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_to_production,
        provide_context=True,
    )

    refresh_baseline_task = PythonOperator(
        task_id='refresh_baseline',
        python_callable=refresh_baseline,
        provide_context=True,
    )

    notify_ok = PythonOperator(
        task_id='notify_approved',
        python_callable=notify_approved,
        provide_context=True,
    )

    trigger_deploy = PythonOperator(
        task_id='trigger_deploy',
        python_callable=trigger_gitlab_pipeline,
        provide_context=True,
    )

    drift_check >> branch
    branch >> no_drift_task
    branch >> alert
    alert >> retrain_model
    retrain_model >> notify_ready_task
    notify_ready_task >> wait_approval
    wait_approval >> promote_model
    promote_model >> refresh_baseline_task >> notify_ok >> trigger_deploy