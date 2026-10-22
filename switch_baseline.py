import argparse
import json
import os
from io import BytesIO

from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "traffic-logs"
POINTER_OBJECT = "baseline_current.txt"


def get_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=False)


def list_baseline_metas(client):
    objects = client.list_objects(BUCKET, prefix="baseline_", recursive=False)
    return [o.object_name for o in objects if o.object_name.endswith(".meta.json")]


def find_by_run_id(client, run_id):
    for meta_name in list_baseline_metas(client):
        resp = client.get_object(BUCKET, meta_name)
        meta = json.loads(resp.read())
        resp.close()
        resp.release_conn()
        if meta.get("mlflow_run_id") == run_id:
            return meta["version"] + ".csv"
    return None


def find_by_tag(client, tag):
    candidate = f"baseline_{tag}.csv"
    try:
        client.stat_object(BUCKET, candidate)
        return candidate
    except Exception:
        return None


def switch_pointer(client, object_name):
    data = object_name.encode("utf-8")
    client.put_object(BUCKET, POINTER_OBJECT, BytesIO(data), len(data))
    print(f"Pointer -> {object_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if not args.run_id and not args.tag:
        parser.error("Нужен --run-id или --tag")

    client = get_client()

    if args.run_id:
        target = find_by_run_id(client, args.run_id)
        if not target:
            raise SystemExit(f"Baseline с run_id={args.run_id} не найден")
    else:
        target = find_by_tag(client, args.tag)
        if not target:
            raise SystemExit(f"Baseline с tag={args.tag} не найден")

    switch_pointer(client, target)


if __name__ == "__main__":
    main()