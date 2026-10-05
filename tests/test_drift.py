import numpy as np
import pandas as pd


def test_psi_zero_when_distributions_identical():
    from dags.scripts.drift_monitor import psi
    data = pd.Series(np.random.randn(1000))
    assert abs(psi(data, data)) < 0.01


def test_psi_high_when_distributions_shifted():
    from dags.scripts.drift_monitor import psi
    baseline = pd.Series(np.random.randn(1000))
    shifted = pd.Series(np.random.randn(1000) + 5)
    assert psi(baseline, shifted) > 0.2