import os
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.read_csv("creditcard.csv")
df.drop_duplicates(inplace=True)
df['Hour'] = (df['Time'] // 3600) % 24
df.drop('Time', axis=1, inplace=True)
df['transaction_id'] = range(1, len(df) + 1)
df['event_timestamp'] = datetime(2026, 1, 1) + pd.to_timedelta(np.arange(len(df)), unit='s')
df['created'] = df['event_timestamp']

FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
features_df = df[['transaction_id', 'event_timestamp', 'created'] + FEATURE_COLS].copy()

targets_df = df[['transaction_id', 'Class']].copy()
targets_df['target_timestamp'] = df['event_timestamp'] + pd.Timedelta(days=3)
targets_df['created'] = targets_df['target_timestamp']

os.makedirs("feature_repo/data", exist_ok=True)

pq.write_table(
    pa.Table.from_pandas(features_df),
    "feature_repo/data/features.parquet"
)
pq.write_table(
    pa.Table.from_pandas(targets_df),
    "feature_repo/data/targets.parquet"
)

print(f"Features: {len(features_df)} rows, columns: {list(features_df.columns)}")
print(f"Targets:  {len(targets_df)} rows, columns: {list(targets_df.columns)}")