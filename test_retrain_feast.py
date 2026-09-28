from feast import FeatureStore
import pandas as pd


store = FeatureStore(repo_path="feature_repo")

df_features_raw = pd.read_parquet("feature_repo/data/features.parquet")
entity_df_features = df_features_raw[["transaction_id", "event_timestamp"]].copy()

fv = store.get_feature_view("creditcard_features")
refs = [f"{fv.name}:{f.name}" for f in fv.features]
features_df = store.get_historical_features(
    entity_df=entity_df_features, features=refs
).to_df()

df_targets_raw = pd.read_parquet("feature_repo/data/targets.parquet")
entity_df_targets = df_targets_raw[["transaction_id", "target_timestamp"]].copy()

fv_t = store.get_feature_view("creditcard_targets")
refs_t = [f"{fv_t.name}:{f.name}" for f in fv_t.features]
targets_df = store.get_historical_features(
    entity_df=entity_df_targets, features=refs_t
).to_df()

merged = features_df.merge(
    targets_df[["transaction_id", "Class"]],
    on="transaction_id",
    how="inner",
)

print(f"features_df rows: {len(features_df)}, columns: {len(features_df.columns)}")
print(f"targets_df rows:  {len(targets_df)}, columns: {len(targets_df.columns)}")
print(f"Class in features: {'Class' in features_df.columns}")
print(f"targets columns: {list(targets_df.columns)}")
print(f"merged shape: {merged.shape}")
print(f"Class in merged: {'Class' in merged.columns}")
print(merged[["transaction_id", "Amount", "Hour", "Class"]].head(3))