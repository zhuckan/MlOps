from datetime import datetime, timedelta
import subprocess
import os
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.python import PythonSensor
from kubernetes.client import models as k8s

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

GITLAB_URL = os.getenv("GITLAB_URL", "http://host.docker.internal:8088")
GITLAB_PROJECT_ID = os.getenv("GITLAB_PROJECT_ID", "1")
GITLAB_TRIGGER_TOKEN = os.getenv("GITLAB_TRIGGER_TOKEN", "")

def run_drift_monitor(**context):
    result = subprocess.run(
        ['python3', '/opt/airflow/dags/scripts/drift_monitor.py'],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'MINIO_ENDPOINT': 'minio:9000',
            'PUSHGATEWAY_URL': 'http://host.docker.internal:9091',
            'FEAST_REPO_PATH': '/opt/airflow/feature_repo',
            'CURRENT_WINDOW_HOURS': '168',
        }
    )
    drift_detected = False
    report_url = None
    for line in result.stdout.splitlines():
        if line.startswith('DRIFT_DETECTED='):
            drift_detected = line.split('=')[1] == 'True'
        elif line.startswith('REPORT_URL='):
            report_url = line.split('=')[1]
    context['ti'].xcom_push(key='drift_detected', value=drift_detected)
    context['ti'].xcom_push(key='report_url', value=report_url)
    return result.stdout

def send_alert(**context):
    drift = context['ti'].xcom_pull(task_ids='drift_check', key='drift_detected')
    if drift:
        report_url = context['ti'].xcom_pull(task_ids='drift_check', key='report_url')
        message = f"Data Drift detected! Report: {report_url}"
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message}
            try:
                response = requests.post(url, json=payload, timeout=10)
                print(f"Telegram response: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Failed to send Telegram message: {e}")
        else:
            print("Telegram credentials not set, printing alert:", message)

def should_continue(**context):
    drift = context['ti'].xcom_pull(task_ids='drift_check', key='drift_detected')
    if drift:
        return 'send_alert'
    return 'no_drift'

def no_drift():
    print("No drift detected, skipping retraining.")

def notify_ready(**context):
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment = mlflow.get_experiment_by_name("Creditcard_Model_Tuning_v2")
    if experiment is None:
        print("Experiment not found")
        return
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_status = 'Ready_for_staging'",
        order_by=["end_time DESC"]
    )
    if not runs_df.empty:
        run_id = runs_df.iloc[0]["run_id"]
        message = f"Новая модель готова к апруву! Run ID: {run_id}. Проверьте MLflow и установите тег Production для деплоя."
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message}
            try:
                response = requests.post(url, json=payload, timeout=10)
                print(f"Telegram notification sent: {response.status_code}")
            except Exception as e:
                print(f"Failed to send Telegram message: {e}")
        else:
            print("Telegram credentials not set, would notify:", message)
    else:
        print("No Ready_for_staging model found.")

def approval_sensor(**context):
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment = mlflow.get_experiment_by_name("Creditcard_Model_Tuning_v2")
    if experiment is None:
        return False
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_status = 'Production'"
    )
    return not runs_df.empty

def trigger_gitlab_pipeline(**context):
    if not GITLAB_TRIGGER_TOKEN:
        print("GITLAB_TRIGGER_TOKEN не задан. Пропускаем триггер.")
        return
    url = f"{GITLAB_URL}/api/v4/projects/{GITLAB_PROJECT_ID}/trigger/pipeline"
    payload = {
        "token": GITLAB_TRIGGER_TOKEN,
        "ref": "main",
        "variables[PROD_DEPLOY]": "true"
    }
    try:
        response = requests.post(url, data=payload, timeout=10)
        print(f"GitLab trigger response: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Failed to trigger GitLab pipeline: {e}")

with DAG(
    dag_id='ml_monitoring_continuous_training',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
    tags=['ml', 'monitoring'],
) as dag:

    drift_check = PythonOperator(
        task_id='drift_check',
        python_callable=run_drift_monitor
    )

    branch = BranchPythonOperator(
        task_id='branch_drift',
        python_callable=should_continue,
        provide_context=True
    )

    no_drift = PythonOperator(
        task_id='no_drift',
        python_callable=no_drift
    )

    alert = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert,
        provide_context=True
    )

    retrain_model = KubernetesPodOperator(
        task_id='retrain_model',
        namespace='ml-service',
        image='retrain-job:v7',
        image_pull_policy='Never',
        name='retrain-job',
        cmds=["python", "retrain.py"],
        env_vars={
            'MLFLOW_TRACKING_URI': 'http://host.minikube.internal:5001',
            'FEAST_REPO_PATH': '/app/feature_repo',
            'MLFLOW_ALLOW_FILE_STORE': 'true',
            'MINIO_ENDPOINT': 'host.minikube.internal:9000',
            'MINIO_ACCESS_KEY': 'minioadmin',
            'MINIO_SECRET_KEY': 'minioadmin',
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

    notify_ready = PythonOperator(
        task_id='notify_ready',
        python_callable=notify_ready,
        provide_context=True,
    )

    wait_approval = PythonSensor(
        task_id='wait_for_approval',
        python_callable=approval_sensor,
        poke_interval=60,
        timeout=3600 * 24 * 7,
        mode='poke',
    )

    trigger_deploy = PythonOperator(
        task_id='trigger_deploy',
        python_callable=trigger_gitlab_pipeline,
        provide_context=True,
    )


    drift_check >> branch
    branch >> no_drift
    branch >> alert
    alert >> retrain_model
    retrain_model >> notify_ready
    notify_ready >> wait_approval
    wait_approval >> trigger_deploy