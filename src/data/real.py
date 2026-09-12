from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def load_california(standardize: bool = True):
    bunch = cast(Bunch, fetch_california_housing(return_X_y=False, as_frame=False))
    X = bunch.data.astype(np.float64)
    y = bunch.target.astype(np.float64)
    if standardize:
        X = StandardScaler().fit_transform(X)
        y = y - y.mean()
    return X, y, list(bunch.feature_names)
