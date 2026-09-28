import os
import time
import random
import argparse
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from minio import Minio


API_URL = os.getenv("API_URL", "http://creditcard.local/predict")
CSV_PATH = "creditcard.csv"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "traffic-logs"


def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def load_baseline():
    df = pd.read_csv(CSV_PATH)
    df.drop_duplicates(inplace=True)
    df['Hour'] = (df['Time'] // 3600) % 24
    df.drop('Time', axis=1, inplace=True)
    df['transaction_id'] = range(1, len(df) + 1)
    return df


def upload_partitioned(df, client, partition_dt, prefix):
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    partition = (
        f"{prefix}/{partition_dt.year}/"
        f"{partition_dt.month:02d}/"
        f"{partition_dt.day:02d}/"
        f"{partition_dt.hour:02d}.csv"
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    client.put_object(BUCKET, partition, BytesIO(csv_bytes), len(csv_bytes))
    print(f"Uploaded {len(df)} rows to {BUCKET}/{partition}")


def simulate_requests(seconds, drift_start, partition_dt):
    df = load_baseline()
    current_rows = []
    predictions_rows = []
    start_time = time.time()

    for i in range(seconds):
        time.sleep(1)
        elapsed = time.time() - start_time
        drift_prob = max(0.0, min(0.5, (elapsed - drift_start) / (seconds - drift_start)))

        if random.random() < drift_prob:
            row = df.sample(1, random_state=42 + i).copy()
            for col in [f'V{j}' for j in range(1, 29)]:
                row[col] += np.random.normal(0, 0.5)
            row['Amount'] *= np.random.uniform(2, 5)
        else:
            row = df.sample(1)

        transaction_id = int(row['transaction_id'].iloc[0])

        payload = (
            row.drop(columns=['Class', 'transaction_id'])
            .to_dict(orient='records')[0]
        )

        try:
            response = requests.post(API_URL, json=payload, timeout=5.0)
            response.raise_for_status()
            pred_data = response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Request failed: {e}")
            continue

        current_rows.append(payload)
        predictions_rows.append({
            "transaction_id": transaction_id,
            "pred": pred_data["pred"],
            "prob": pred_data["prob"],
        })

    if not current_rows:
        print("No successful requests, nothing to upload.")
        return

    client = get_minio_client()

    current_df = pd.DataFrame(current_rows)
    upload_partitioned(current_df, client, partition_dt, prefix="current")

    predictions_df = pd.DataFrame(predictions_rows)
    upload_partitioned(predictions_df, client, partition_dt, prefix="predictions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--drift-start", type=int, default=60)
    parser.add_argument("--partition-date", type=str, default=None,
                        help="Override partition date (YYYY-MM-DD).")
    parser.add_argument("--partition-hour", type=int, default=None,
                        help="Override partition hour (0-23).")
    args = parser.parse_args()

    partition_dt = datetime.utcnow()

    if args.partition_date is not None:
        d = datetime.strptime(args.partition_date, "%Y-%m-%d")
        partition_dt = partition_dt.replace(year=d.year, month=d.month, day=d.day)

    if args.partition_hour is not None:
        partition_dt = partition_dt.replace(hour=args.partition_hour)

    print(f"Target partition: {partition_dt}")
    print(f"API URL: {API_URL}")
    simulate_requests(args.seconds, args.drift_start, partition_dt)


if __name__ == "__main__":
    main()