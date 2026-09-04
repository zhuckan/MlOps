import time
import random
import numpy as np
import pandas as pd
import requests
from minio import Minio
from io import BytesIO

API_URL = "http://localhost:8000/predict"
CSV_PATH = "creditcard.csv"
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "traffic-logs"
CURRENT_OBJECT = "current.csv"

def load_baseline():
    df = pd.read_csv(CSV_PATH)
    df.drop_duplicates(inplace=True)
    df['Hour'] = (df['Time'] // 3600) % 24
    df.drop('Time', axis=1, inplace=True)
    return df

def upload_to_minio(df, bucket, object_name):
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    client.put_object(bucket, object_name, BytesIO(csv_bytes), len(csv_bytes))

def simulate_requests(seconds=300, drift_start=60):
    df = load_baseline()
    current_rows = []
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

        payload = row.drop('Class', axis=1).to_dict(orient='records')[0]
        requests.post(API_URL, json=payload, timeout=5.0)
        current_rows.append(row.drop('Class', axis=1).iloc[0].to_dict())

    current_df = pd.DataFrame(current_rows)
    upload_to_minio(current_df, BUCKET, CURRENT_OBJECT)

if __name__ == "__main__":
    simulate_requests()