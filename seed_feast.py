import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import os

df = pd.read_csv("creditcard.csv")
df.drop_duplicates(inplace=True)
df['Hour'] = (df['Time'] // 3600) % 24
df.drop('Time', axis=1, inplace=True)

df['transaction_id'] = range(1, len(df) + 1)
df['event_timestamp'] = datetime(2026, 1, 1) + pd.to_timedelta(np.arange(len(df)), unit='s')
df['created'] = df['event_timestamp']

os.makedirs("feature_repo/data", exist_ok=True)
table = pa.Table.from_pandas(df)
pq.write_table(table, "feature_repo/data/features.parquet")
print("Features parquet saved.")