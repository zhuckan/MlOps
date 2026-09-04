from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

transaction = Entity(name="transaction_id", join_keys=["transaction_id"])

source = FileSource(
    name="creditcard_source",
    path="data/features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

feature_view = FeatureView(
    name="creditcard_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="V1", dtype=Float32),
        Field(name="V2", dtype=Float32),
        Field(name="V3", dtype=Float32),
        Field(name="V4", dtype=Float32),
        Field(name="V5", dtype=Float32),
        Field(name="V6", dtype=Float32),
        Field(name="V7", dtype=Float32),
        Field(name="V8", dtype=Float32),
        Field(name="V9", dtype=Float32),
        Field(name="V10", dtype=Float32),
        Field(name="V11", dtype=Float32),
        Field(name="V12", dtype=Float32),
        Field(name="V13", dtype=Float32),
        Field(name="V14", dtype=Float32),
        Field(name="V15", dtype=Float32),
        Field(name="V16", dtype=Float32),
        Field(name="V17", dtype=Float32),
        Field(name="V18", dtype=Float32),
        Field(name="V19", dtype=Float32),
        Field(name="V20", dtype=Float32),
        Field(name="V21", dtype=Float32),
        Field(name="V22", dtype=Float32),
        Field(name="V23", dtype=Float32),
        Field(name="V24", dtype=Float32),
        Field(name="V25", dtype=Float32),
        Field(name="V26", dtype=Float32),
        Field(name="V27", dtype=Float32),
        Field(name="V28", dtype=Float32),
        Field(name="Amount", dtype=Float32),
        Field(name="Hour", dtype=Int64),
        Field(name="Class", dtype=Int64),
    ],
    source=source,
)