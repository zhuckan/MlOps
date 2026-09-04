from datetime import datetime, timedelta
import subprocess
import os
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sensors.python import PythonSensor

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_drift_monitor(**context):
    result = subprocess.run(
        ['python3', '/opt/airflow/dags/scripts/drift_monitor.py'],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'MINIO_ENDPOINT': 'minio:9000',
            'PUSHGATEWAY_URL': 'http://host.docker.internal:9091'
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

with DAG(
    dag_id='ml_monitoring_continuous_training',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
    tags=['ml', 'monitoring'],
) as dag:
    drift_check = PythonOperator(task_id='drift_check', python_callable=run_drift_monitor)
    branch = BranchPythonOperator(task_id='branch_drift', python_callable=should_continue, provide_context=True)
    no_drift = PythonOperator(task_id='no_drift', python_callable=no_drift)
    alert = PythonOperator(task_id='send_alert', python_callable=send_alert, provide_context=True)

    retrain_model = KubernetesPodOperator(
        task_id='retrain_model',
        namespace='ml-service',
        image='retrain-job:v3',
        image_pull_policy='Never',
        name='retrain-job',
        cmds=["python", "retrain.py"],
        env_vars={
            'MLFLOW_TRACKING_URI': 'http://host.docker.internal:5001',
            'FEAST_REPO_PATH': '/app/feature_repo',
            'MLFLOW_ALLOW_FILE_STORE': 'true',
        },
        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=False,
        config_file="/opt/airflow/kubeconfig-ci.yaml",
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
        python_callable=lambda: print("Trigger deploy pipeline"),
    )

    drift_check >> branch
    branch >> no_drift
    branch >> alert
    alert >> retrain_model
    retrain_model >> wait_approval >> trigger_deploy