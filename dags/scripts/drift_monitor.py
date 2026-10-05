import io
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from minio import Minio
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")

BUCKET_TRAFFIC = "traffic-logs"
BUCKET_REPORTS = "drift-reports"
POINTER_OBJECT = "baseline_current.txt"
CURRENT_PREFIX = "current/"
PREDICTIONS_PREFIX = "predictions/"

PUSHGATEWAY_URL = os.getenv(
    "PUSHGATEWAY_URL",
    "http://pushgateway.monitoring.svc.cluster.local:9091",
)

PSI_THRESHOLD = 0.2
KS_ALPHA = 0.05
CURRENT_WINDOW_HOURS = int(os.getenv("CURRENT_WINDOW_HOURS", "24"))
CONCEPT_AVG_PROB_HIGH = 0.5
CONCEPT_AVG_PROB_LOW = 0.0001
CONCEPT_ERROR_RATE_HIGH = 0.3


def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def read_object(client, bucket, object_name):
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def read_baseline(client):
    pointer_bytes = read_object(client, BUCKET_TRAFFIC, POINTER_OBJECT)
    baseline_name = pointer_bytes.decode("utf-8").strip()
    print(f"Baseline pointer -> {baseline_name}")

    csv_bytes = read_object(client, BUCKET_TRAFFIC, baseline_name)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"Loaded baseline: {baseline_name} ({len(df)} rows)")
    return df, baseline_name


def read_partitioned_window(client, prefix, hours):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    objects = client.list_objects(BUCKET_TRAFFIC, prefix=prefix, recursive=True)

    frames = []
    for obj in objects:
        path = obj.object_name
        try:
            parts = path.replace(prefix, "").replace(".csv", "").split("/")
            year, month, day, hour = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            obj_dt = datetime(year, month, day, hour)
        except (ValueError, IndexError):
            continue

        if obj_dt < cutoff:
            continue

        csv_bytes = read_object(client, BUCKET_TRAFFIC, path)
        frames.append(pd.read_csv(io.BytesIO(csv_bytes)))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(frames)} partitions ({prefix}), {len(combined)} rows")
    return combined


def read_targets(feast_repo_path):
    targets_path = os.path.join(feast_repo_path, "data", "targets.parquet")
    if not os.path.exists(targets_path):
        print(f"Targets file not found: {targets_path}")
        return pd.DataFrame()

    df = pd.read_parquet(targets_path)
    now = pd.Timestamp.utcnow().tz_localize(None)
    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

    available = df[df["target_timestamp"] <= now].copy()
    print(f"Targets available: {len(available)} of {len(df)} (labels come with delay)")
    return available


def psi(expected: pd.Series, actual: pd.Series, buckets=10):
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breaks = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breaks = np.unique(breaks)
    if len(breaks) < 2:
        return 0.0

    expected_counts = pd.cut(expected, breaks, include_lowest=True).value_counts().sort_index()
    actual_counts = pd.cut(actual, breaks, include_lowest=True).value_counts().sort_index()

    expected_ratio = expected_counts / expected_counts.sum()
    actual_ratio = actual_counts / actual_counts.sum()

    epsilon = 1e-10
    return float(np.sum(
        (actual_ratio - expected_ratio)
        * np.log((actual_ratio + epsilon) / (expected_ratio + epsilon))
    ))


def compute_data_drift_metrics(baseline_df, current_df):
    features = [c for c in baseline_df.columns if c != "Class"]
    n_features = len(features)
    ks_threshold = KS_ALPHA / n_features

    metrics = {}
    drift_detected = False

    for col in features:
        if col not in current_df.columns:
            continue

        psi_val = psi(baseline_df[col], current_df[col])
        ks_stat, ks_pvalue = ks_2samp(baseline_df[col], current_df[col])

        metrics[f"{col}_psi"] = psi_val
        metrics[f"{col}_ks_pvalue"] = ks_pvalue

        if psi_val > PSI_THRESHOLD or ks_pvalue < ks_threshold:
            drift_detected = True

    return metrics, drift_detected


def compute_concept_drift_metrics(predictions_df, targets_df):
    metrics = {}
    drift_detected = False

    if predictions_df.empty:
        return metrics, drift_detected

    avg_prob = float(predictions_df["prob"].mean())
    max_prob = float(predictions_df["prob"].max())
    pred_rate = float((predictions_df["pred"] == 1).mean())

    metrics["concept_avg_prob"] = avg_prob
    metrics["concept_max_prob"] = max_prob
    metrics["concept_pred_rate"] = pred_rate
    metrics["concept_predictions_samples"] = float(len(predictions_df))

    if avg_prob > CONCEPT_AVG_PROB_HIGH or avg_prob < CONCEPT_AVG_PROB_LOW:
        drift_detected = True

    if not targets_df.empty and "transaction_id" in predictions_df.columns:
        joined = predictions_df.merge(
            targets_df[["transaction_id", "Class"]],
            on="transaction_id",
            how="inner",
        )
        if not joined.empty:
            error_rate = float((joined["pred"] != joined["Class"]).mean())
            metrics["concept_error_rate"] = error_rate
            metrics["concept_labeled_samples"] = float(len(joined))
            if error_rate > CONCEPT_ERROR_RATE_HIGH:
                drift_detected = True
        else:
            metrics["concept_error_rate"] = 0.0
            metrics["concept_labeled_samples"] = 0.0
    else:
        metrics["concept_error_rate"] = 0.0
        metrics["concept_labeled_samples"] = 0.0

    return metrics, drift_detected


def push_metrics(metrics):
    registry = CollectorRegistry()
    for name, value in metrics.items():
        Gauge(f"drift_{name}", f"Drift metric {name}", registry=registry).set(value)
    push_to_gateway(PUSHGATEWAY_URL, job="drift_monitor", registry=registry)
    print(f"Pushed {len(metrics)} metrics to {PUSHGATEWAY_URL}")


def _render_data_drift_table(metrics):
    rows = []
    feature_names = set()
    for key in metrics:
        if key.endswith("_psi"):
            feature_names.add(key[:-4])
        elif key.endswith("_ks_pvalue"):
            feature_names.add(key[:-10])

    ks_threshold = KS_ALPHA / len(feature_names) if feature_names else KS_ALPHA

    for feat in sorted(feature_names):
        psi_val = metrics.get(f"{feat}_psi")
        ks_val = metrics.get(f"{feat}_ks_pvalue")
        is_drift = (psi_val and psi_val > PSI_THRESHOLD) or (ks_val and ks_val < ks_threshold)
        rows.append({
            "Feature": feat,
            "PSI": f"{psi_val:.4f}" if psi_val is not None else "N/A",
            "KS p-value": f"{ks_val:.6f}" if ks_val is not None else "N/A",
            "Drift": "YES" if is_drift else "no",
        })

    return pd.DataFrame(rows).to_html(index=False, escape=False, classes="table")


def _render_concept_drift_table(metrics):
    if not metrics:
        return "<p>No prediction data available.</p>"

    rows = []
    avg_prob = metrics.get("concept_avg_prob", 0.0)
    rows.append({
        "Metric": "avg_prob",
        "Value": f"{avg_prob:.6f}",
        "Threshold": f"[{CONCEPT_AVG_PROB_LOW}, {CONCEPT_AVG_PROB_HIGH}]",
        "Drift": "YES" if avg_prob > CONCEPT_AVG_PROB_HIGH or avg_prob < CONCEPT_AVG_PROB_LOW else "no",
    })
    rows.append({
        "Metric": "max_prob",
        "Value": f"{metrics.get('concept_max_prob', 0.0):.6f}",
        "Threshold": "-",
        "Drift": "-",
    })
    rows.append({
        "Metric": "pred_rate",
        "Value": f"{metrics.get('concept_pred_rate', 0.0):.6f}",
        "Threshold": "-",
        "Drift": "-",
    })
    rows.append({
        "Metric": "predictions_samples",
        "Value": f"{int(metrics.get('concept_predictions_samples', 0))}",
        "Threshold": "-",
        "Drift": "-",
    })

    err = metrics.get("concept_error_rate", 0.0)
    rows.append({
        "Metric": "error_rate",
        "Value": f"{err:.6f}",
        "Threshold": f"<= {CONCEPT_ERROR_RATE_HIGH}",
        "Drift": "YES" if err > CONCEPT_ERROR_RATE_HIGH else "no",
    })
    rows.append({
        "Metric": "labeled_samples",
        "Value": f"{int(metrics.get('concept_labeled_samples', 0))}",
        "Threshold": "-",
        "Drift": "-",
    })

    return pd.DataFrame(rows).to_html(index=False, escape=False, classes="table")


def generate_html_report(data_metrics, concept_metrics, drift_detected, baseline_name):
    data_table = _render_data_drift_table(data_metrics)
    concept_table = _render_concept_drift_table(concept_metrics)

    status = "DRIFT DETECTED" if drift_detected else "OK"
    feature_names = {k[:-4] for k in data_metrics if k.endswith("_psi")}
    ks_threshold = KS_ALPHA / len(feature_names) if feature_names else KS_ALPHA

    return f"""<!DOCTYPE html>
<html><head><title>Drift Report</title>
<style>
 body {{ font-family: Arial; margin: 20px; }}
 h1, h2 {{ color: #333; }}
 .summary {{ font-size: 16px; margin-bottom: 20px; }}
 .table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
 .table th, .table td {{ border: 1px solid #ddd; padding: 6px; }}
 .table th {{ background: #f2f2f2; }}
</style></head><body>
<h1>Drift Report</h1>
<div class="summary">
 <strong>{status}</strong><br>
 Baseline: {baseline_name}<br>
 PSI threshold: {PSI_THRESHOLD}<br>
 KS alpha / Bonferroni: {KS_ALPHA} / {ks_threshold:.6f}<br>
 Error rate threshold: {CONCEPT_ERROR_RATE_HIGH}<br>
 Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
</div>
<h2>Data Drift</h2>
{data_table}
<h2>Concept Drift (predictions &amp; delayed labels)</h2>
{concept_table}
</body></html>"""


def upload_report(client, html_report):
    if not client.bucket_exists(BUCKET_REPORTS):
        client.make_bucket(BUCKET_REPORTS)

    report_name = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    html_bytes = html_report.encode("utf-8")
    client.put_object(BUCKET_REPORTS, report_name, io.BytesIO(html_bytes), len(html_bytes))
    return f"http://{MINIO_ENDPOINT}/{BUCKET_REPORTS}/{report_name}"


def _write_xcom_result(drift_detected, report_url):
    xcom_path = "/airflow/xcom/return.json"
    try:
        with open(xcom_path, "w") as f:
            json.dump({
                "drift_detected": bool(drift_detected),
                "report_url": report_url or "",
            }, f)
        print(f"[xcom] wrote result to {xcom_path}")
    except (FileNotFoundError, PermissionError):
        pass


def main():
    client = get_minio_client()
    baseline_df, baseline_name = read_baseline(client)
    current_df = read_partitioned_window(client, CURRENT_PREFIX, CURRENT_WINDOW_HOURS)
    predictions_df = read_partitioned_window(client, PREDICTIONS_PREFIX, CURRENT_WINDOW_HOURS)
    targets_df = read_targets(FEAST_REPO_PATH)

    if current_df.empty:
        print("DRIFT_DETECTED=False")
        print("REPORT_URL=none")
        _write_xcom_result(False, "none")
        return

    data_metrics, data_drift = compute_data_drift_metrics(baseline_df, current_df)
    concept_metrics, concept_drift = compute_concept_drift_metrics(predictions_df, targets_df)

    all_metrics = {**data_metrics, **concept_metrics}
    drift_detected = data_drift or concept_drift

    try:
        push_metrics(all_metrics)
    except Exception as e:
        print(f"[warn] push_metrics failed: {e}")

    html_report = generate_html_report(data_metrics, concept_metrics, drift_detected, baseline_name)
    report_url = upload_report(client, html_report)

    print(f"DRIFT_DETECTED={'True' if drift_detected else 'False'}")
    print(f"REPORT_URL={report_url}")
    _write_xcom_result(drift_detected, report_url)


if __name__ == "__main__":
    main()