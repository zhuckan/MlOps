import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from minio import Minio
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_TRAFFIC = "traffic-logs"
BUCKET_REPORTS = "drift-reports"
BASELINE_OBJECT = "baseline.csv"
CURRENT_OBJECT = "current.csv"
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")

PSI_THRESHOLD = 0.0
KS_PVALUE_THRESHOLD = 0.05

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def read_csv_from_minio(client, bucket, object_name):
    response = client.get_object(bucket, object_name)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    return df

def psi(expected: pd.Series, actual: pd.Series, buckets=10):
    expected = expected.dropna()
    actual = actual.dropna()
    breaks = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts = pd.cut(expected, breaks, include_lowest=True).value_counts().sort_index()
    actual_counts = pd.cut(actual, breaks, include_lowest=True).value_counts().sort_index()
    expected_ratio = expected_counts / expected_counts.sum()
    actual_ratio = actual_counts / actual_counts.sum()
    epsilon = 1e-10
    return float(np.sum((actual_ratio - expected_ratio) * np.log((actual_ratio + epsilon) / (expected_ratio + epsilon))))

def compute_drift_metrics(baseline_df, current_df):
    features = [c for c in baseline_df.columns if c != 'Class']
    metrics = {}
    drift_detected = False
    for col in features:
        psi_val = psi(baseline_df[col], current_df[col])
        ks_stat, ks_pvalue = ks_2samp(baseline_df[col], current_df[col])
        metrics[f"{col}_psi"] = psi_val
        metrics[f"{col}_ks_pvalue"] = ks_pvalue
        if psi_val > PSI_THRESHOLD or ks_pvalue < KS_PVALUE_THRESHOLD:
            drift_detected = True
    return metrics, drift_detected

def push_metrics(metrics):
    registry = CollectorRegistry()
    for name, value in metrics.items():
        Gauge(f"drift_{name}", f"Drift metric {name}", registry=registry).set(value)
    push_to_gateway(PUSHGATEWAY_URL, job="drift_monitor", registry=registry)

def generate_html_report(metrics, drift_detected):
    rows = []
    feature_names = set()
    for key in metrics:
        if key.endswith('_psi'):
            feature_names.add(key[:-4])
        elif key.endswith('_ks_pvalue'):
            feature_names.add(key[:-10])
    for feat in sorted(feature_names):
        psi_val = metrics.get(f"{feat}_psi")
        ks_val = metrics.get(f"{feat}_ks_pvalue")
        rows.append({
            "Feature": feat,
            "PSI": f"{psi_val:.4f}" if psi_val is not None else "N/A",
            "KS p-value": f"{ks_val:.4f}" if ks_val is not None else "N/A",
            "Drift": "YES" if (psi_val and psi_val > PSI_THRESHOLD) or (ks_val and ks_val < KS_PVALUE_THRESHOLD) else "no"
        })
    df = pd.DataFrame(rows)
    html_table = df.to_html(index=False, escape=False, classes='table table-striped')
    html = f"""<!DOCTYPE html>
<html>
<head><title>Data Drift Report</title>
<style>
    body {{ font-family: Arial; margin: 20px; }}
    h1 {{ color: #333; }}
    .summary {{ font-size: 18px; margin-bottom: 20px; }}
    .table {{ border-collapse: collapse; width: 100%; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    .table th {{ background-color: #f2f2f2; }}
    .drift-yes {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>Data Drift Monitoring Report</h1>
<div class="summary">Drift detected: <strong class="{('drift-yes' if drift_detected else '')}">{str(drift_detected).upper()}</strong><br>
Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
{html_table}
</body></html>"""
    return html

def upload_report_to_minio(html_report):
    client = get_minio_client()
    if not client.bucket_exists(BUCKET_REPORTS):
        client.make_bucket(BUCKET_REPORTS)
    report_name = f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    html_bytes = html_report.encode('utf-8')
    client.put_object(BUCKET_REPORTS, report_name, io.BytesIO(html_bytes), len(html_bytes))
    return f"http://{MINIO_ENDPOINT}/{BUCKET_REPORTS}/{report_name}"

def main():
    client = get_minio_client()
    baseline_df = read_csv_from_minio(client, BUCKET_TRAFFIC, BASELINE_OBJECT)
    current_df = read_csv_from_minio(client, BUCKET_TRAFFIC, CURRENT_OBJECT)
    metrics, drift_detected = compute_drift_metrics(baseline_df, current_df)
    # push_metrics(metrics)
    html_report = generate_html_report(metrics, drift_detected)
    report_url = upload_report_to_minio(html_report)
    print(f"DRIFT_DETECTED={'True' if drift_detected else 'False'}")
    print(f"REPORT_URL={report_url}")

if __name__ == "__main__":
    main()