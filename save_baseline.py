import os
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
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def create_baseline_version(tag):
    df = pd.read_csv(CSV_PATH)
    df.drop_duplicates(inplace=True)
    df['Hour'] = (df['Time'] // 3600) % 24
    df.drop('Time', axis=1, inplace=True)
    df.drop('Class', axis=1, inplace=True)

    if tag is None:
        tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    object_name = f"baseline_{tag}.csv"

    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    client.put_object(BUCKET, object_name, BytesIO(csv_bytes), len(csv_bytes))
    print(f"Baseline uploaded: {BUCKET}/{object_name} ({len(df)} rows)")

    pointer_bytes = object_name.encode("utf-8")
    client.put_object(BUCKET, POINTER_OBJECT, BytesIO(pointer_bytes), len(pointer_bytes))
    print(f"Pointer updated: {BUCKET}/{POINTER_OBJECT} → {object_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None,
                        help="Version tag (default: current UTC timestamp)")
    args = parser.parse_args()
    create_baseline_version(args.tag)


if __name__ == "__main__":
    main()