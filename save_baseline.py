import pandas as pd
from minio import Minio
from io import BytesIO

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "traffic-logs"
BASELINE_OBJECT = "baseline.csv"

# Читаем исходный датасет
df = pd.read_csv("creditcard.csv")
df.drop_duplicates(inplace=True)
df['Hour'] = (df['Time'] // 3600) % 24
df.drop('Time', axis=1, inplace=True)

df.drop('Class', axis=1, inplace=True)

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)
if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

csv_bytes = df.to_csv(index=False).encode()
client.put_object(BUCKET, BASELINE_OBJECT, BytesIO(csv_bytes), len(csv_bytes))
print(f"Baseline uploaded to {BUCKET}/{BASELINE_OBJECT}")