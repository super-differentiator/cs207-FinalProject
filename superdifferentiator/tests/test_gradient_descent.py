from superdifferentiator.forward.functions import X
from superdifferentiator.additional_features.gradient_descent import GD
import pytest


def test_create_GD():
    p = GD(max_iter=100, precision=0.1, step_size=0.01)
    assert (p.max_iter == 100)
    assert (p.step_size == 0.01)
    assert (p.precision == 0.1)


def test_cal_gradient_1d():
    def foo(x):
        fx = x ** 2 + x
        return fx.val, fx.der

    p = GD(max_iter=10000, step_size=0.001, precision=0.0000001)
    x = X(0)
    val = p.cal_gradient_1d(foo, x)
    assert (abs(-0.5 - val) < 0.01)


def test_cal_gradient_2d():
    def foo(x, y):
        fx = x ** 2 + y ** 2
        return fx.val, fx.der

    p = GD(max_iter=10000, step_size=0.01, precision=0.0001)
    x = X(10, 'x')
    y = X(10, 'y')
    val = p.cal_gradient_2d(foo, x, y)
    assert (abs(0 - val[0][0]) < 0.01)
    assert (abs(0 - val[1][0]) < 0.01)


def test_cal_gradient_3d():
    def foo(x, y, z):
        fx = x ** 2 + y ** 2 + z ** 2
        return fx.val, fx.der

    p = GD(max_iter=10000, step_size=0.01, precision=0.0001)
    x = X(5, 'x')
    y = X(5, 'y')
    z = X(5, 'z')
    val = p.cal_gradient_3d(foo, x, y, z)
    assert (abs(0 - val[0][0]) < 0.01)
    assert (abs(0 - val[1][0]) < 0.01)
    assert (abs(0 - val[2][0]) < 0.01)


def test_cal_gradient_nd():
    def foo(vars):
        fx = vars[0] ** 2 + vars[1] ** 2 + vars[2] ** 2 + vars[3] ** 2
        return fx.val, fx.der

    x = X(10, 'x')
    y = X(10, 'y')
    z = X(10, 'z')
    a = X(10, 'a')

    p = GD(max_iter=10000, step_size=0.01, precision=0.0001)
    val = p.cal_gradient_nd(foo, [x, y, z, a])
    assert (abs(0 - val[0][0]) < 0.01)
    assert (abs(0 - val[1][0]) < 0.01)
    assert (abs(0 - val[2][0]) < 0.01)
    assert (abs(0 - val[3][0]) < 0.01)
