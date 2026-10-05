import pandas as pd
import pytest


def test_temporal_split_keeps_order():
    from retrain import temporal_split
    df = pd.DataFrame({
        "event_timestamp": pd.to_datetime([
            "2026-01-04", "2026-01-01", "2026-01-03", "2026-01-02",
        ]),
        "Class": [0, 1, 0, 0],
    })
    train, holdout = temporal_split(df, holdout_fraction=0.5)
    assert train["event_timestamp"].max() < holdout["event_timestamp"].min()


def test_temporal_split_raises_on_small_data():
    from retrain import temporal_split
    df = pd.DataFrame({
        "event_timestamp": pd.to_datetime(["2026-01-01"]),
        "Class": [0],
    })
    with pytest.raises(ValueError):
        temporal_split(df, holdout_fraction=0.5)


def test_temporal_split_raises_on_single_class():
    from retrain import temporal_split
    df = pd.DataFrame({
        "event_timestamp": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        ]),
        "Class": [1, 1, 1, 1],
    })
    with pytest.raises(ValueError):
        temporal_split(df, holdout_fraction=0.5)