"""Real-world regression dataset loader (California Housing from sklearn)."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def load_california(standardize: bool = True):
    """Returns (X, y, feature_names). Standardized by default for nicer optimization."""
    bunch = fetch_california_housing()
    X = bunch.data.astype(np.float64)
    y = bunch.target.astype(np.float64)
    if standardize:
        X = StandardScaler().fit_transform(X)
        # also center y to remove a huge intercept
        y = y - y.mean()
    return X, y, list(bunch.feature_names)
