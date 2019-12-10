from superdifferentiator.forward.functions import X
from superdifferentiator.additional_features.linear_regression import LinearRegression
import pytest
import numpy as np


def test_create_linear_regression():
    clf = LinearRegression(max_iter=10000)
    assert (clf.max_iter==10000)
    assert (clf.coef_ is None)


def test_linear_regression():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 5])
    clf = LinearRegression(max_iter=10000)
    clf.fit(x, y)
    sc = clf.score(x, y)
    assert (abs(sc - 0.96) <= 1)