import pytest
import numpy as np

from ..AD import X, Cos, Sin, Tan, Log, Exp


def test_create_x():
    x = X(2)
    assert x.val == 2
    assert x.der == 1


def test_create_cos():
    x = X(2)
    x1 = Cos(x)
    assert x1.val == np.cos(2)
    assert x1.der == -np.sin(2)


def test_create_sin():
    x = X(2)
    x1 = Sin(x)
    assert x1.val == np.sin(2)
    assert x1.der == np.cos(2)


def test_create_tan():
    x = X(2)
    x1 = Tan(x)
    assert x1.val == np.tan(2)
    assert x1.der == 1 / np.cos(2) ** 2


def test_create_log():
    x = X(2)
    x1 = Log(x)
    assert x1.val == np.log(2)
    assert x1.der == 0.5


def test_create_exp():
    x = X(2)
    x1 = Exp(x)
    assert x1.val == np.exp(2)
    assert x1.der == np.exp(2)

def test_neg():
    x = X(2)
    x1 = -x
    assert x1.val == -2
    assert x1.der == -1

def test_add():
    x = X(2)
    f = x + 3

    assert f.val == 5
    assert f.der == 1


def test_add1():
    x = X(2)
    f = x + x
    assert f.val == 4
    assert f.der == 2


def test_radd():
    x = X(2)
    f = 3 + x
    assert f.val == 5
    assert f.der == 1


def test_sub():
    x = X(2)
    f = x - 3
    assert f.val == -1
    assert f.der == 1


def test_sub1():
    x = X(2)
    f = x - x
    assert f.val == 0
    assert f.der == 0


def test_rsub():
    x = X(2)
    f = 3 - x
    assert f.val == 1
    assert f.der == -1


def test_mul():
    x = X(2)
    f = x * 3
    assert f.val == 6
    assert f.der == 3


def test_mul1():
    x = X(2)
    f = x * x
    assert f.val == 4
    assert f.der == 4


def test_rmul():
    x = X(2)
    f = 3 * x
    assert f.val == 6
    assert f.der == 3


def test_div():
    x = X(2)
    f = x / 2
    assert f.val == 1
    assert f.der == 0.5


def test_div1():
    x = X(2)
    f = x / x
    assert f.val == 1
    assert f.der == 1


def test_rdiv():
    x = X(2)
    f = 2 / x
    assert f.val == 1
    assert f.der == -0.5


def test_pow():
    x = X(2)
    f = x ** 3
    assert f.val == 8
    assert f.der == 12


def test_rpow():
    x = X(2)
    f = 3 ** x
    assert f.val == 9
    assert f.der == 9 * np.log(3)


def add_invalid():
    x = X(2)
    f = x + 'hello'


def test_invalid_add():
    with pytest.raises(ValueError) as excinfo:
        add_invalid()
    assert 'operand in addition is invalid' == str(excinfo.value)


def mul_invalid():
    x = X(2)
    f = x * 'hello'


def test_invalid_mul():
    with pytest.raises(ValueError) as excinfo:
        mul_invalid()
    assert 'operand in multiplication is invalid' == str(excinfo.value)
