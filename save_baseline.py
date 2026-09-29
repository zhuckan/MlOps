import os
import json
import argparse
from datetime import datetime
from io import BytesIO

import pandas as pd
from minio import Minio


CSV_PATH = "creditcard.csv"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "traffic-logs"
POINTER_OBJECT = "baseline_current.txt"


def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=False)


def create_baseline_version(tag, mlflow_run_id=None):
    df = pd.read_csv(CSV_PATH)
    df.drop_duplicates(inplace=True)
    df['Hour'] = (df['Time'] // 3600) % 24
    df.drop('Time', axis=1, inplace=True)
    df.drop('Class', axis=1, inplace=True)

    if tag is None:
        tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    object_name = f"baseline_{tag}.csv"
    meta_name = f"baseline_{tag}.meta.json"

    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    client.put_object(BUCKET, object_name, BytesIO(csv_bytes), len(csv_bytes))

    meta = {
        "version": f"baseline_{tag}",
        "created_at": datetime.utcnow().isoformat(),
        "rows": len(df),
        "mlflow_run_id": mlflow_run_id,
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    client.put_object(BUCKET, meta_name, BytesIO(meta_bytes), len(meta_bytes))

    pointer_bytes = object_name.encode("utf-8")
    client.put_object(BUCKET, POINTER_OBJECT, BytesIO(pointer_bytes), len(pointer_bytes))

    print(f"Baseline: {object_name} ({len(df)} rows)")
    print(f"Metadata: {meta_name}")
    print(f"MLflow run_id: {mlflow_run_id}")
    print(f"Pointer updated -> {object_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--mlflow-run-id", type=str, default=None)
    args = parser.parse_args()
    create_baseline_version(args.tag, args.mlflow_run_id)


if __name__ == "__main__":
    main()