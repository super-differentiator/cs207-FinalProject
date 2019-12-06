import pytest
import numpy as np

from superdifferentiator.forward.functions import X, Sin, Cos, Tan, Ln, Log, Exp, Abs, Sqrt, Arcsin, Arccos, Arctan, Sinh, Cosh, Tanh, Logistic

def test_create_x():
	x = X(2)
	assert x.val == [2]
	assert x.der['x'] == [1]

def test_multiple_values():
	# f(x) = x^2 + 2x + 3, evaluate at f(3) and f(5)
	x = X([3, 5])
	fx = (x ** 2) + (2 * x) + 3
	assert fx.val == [18, 38]
	assert fx.der['x'] == [8, 12]

def test_multiple_variables():
	# f(x, y) = x^2 * y^3, evaluate at f(3, 4)
	x = X(3, 'x')
	y = X(4, 'y')
	f_xy = (x ** 2) * (y ** 3)
	assert f_xy.val == [576]
	assert f_xy.der['x'] == [384]
	assert f_xy.der['y'] == [432]

def test_multiple_variables_and_values():
	# f(x, y) = 2xy^2, evaluate at f(8, 9) and f(4, 12)
	x = X([8, 4], 'x')
	y = X([9, 12], 'y')
	f_xy = 2 * x * (y ** 2)
	assert f_xy.val == [1296, 1152]
	assert f_xy.der['x'] == [162, 288]
	assert f_xy.der['y'] == [288, 192]

def test_scalar_jacobian():
	# This is basically just calculating the gradient of f(x, y) = x^2 * y^2
	# since the Jacobian for a scalar function is the gradient.
	x = X(3, 'x')
	y = X(4, 'y')
	fx = x * x * y * y
	allVars, jacs = fx.jacobian()
	assert 'x' in allVars
	assert 'y' in allVars
	
	xIdx = allVars.index('x')
	yIdx = allVars.index('y')
	jac = jacs[0]
	assert jac[0, xIdx] == 96
	assert jac[0, yIdx] == 72

def test_eq():
	x1 = X(3)
	x2 = X(3)
	assert x1 == x2

def test_neq():
	x1 = X(3)
	x2 = X(4)
	assert x1 != x2

def test_create_cos():
	x = X(2)
	x1 = Cos(x)
	assert x1.val == [np.cos(2)]
	assert x1.der['x'] == [-np.sin(2)]

def test_cos2():
	c = Cos(8)
	assert c.val == [np.cos(8)]
	assert c.der['x'] == [0]


def test_create_sin():
	x = X(2)
	x1 = Sin(x)
	assert x1.val == [np.sin(2)]
	assert x1.der['x'] == [np.cos(2)]

def test_sin2():
	s = Sin(5)
	assert s.val == [np.sin(5)]
	assert s.der['x'] == [0]

def test_create_tan():
	x = X(2)
	x1 = Tan(x)
	assert x1.val == [np.tan(2)]
	assert x1.der['x'] == [1 / np.cos(2) ** 2]

def test_tan2():
	t = Tan(11)
	assert t.val == [np.tan(11)]
	assert t.der['x'] == [0]

def test_create_ln():
	x = X(2)
	x1 = Ln(x)
	assert x1.val == [np.log(2)]
	assert x1.der['x'] == [0.5]

def test_ln2():
	ln = Ln(5)
	assert ln.val == [ np.log(5)]
	assert ln.der['x'] == [0]

def test_create_log():
	x = X(2)
	log = Log(x, 10)
	assert log.val == [np.log(2) / np.log(10)]
	assert log.der['x'] == [1 / (2 * np.log(10))]

def test_log2():
	log = Log(100, 10)

	# Had to use approx since there was some floating point error on this one
	assert log.val == [pytest.approx(np.log(100) / np.log(10))]
	assert log.der['x'] == [0]

def test_create_exp():
	x = X(2)
	x1 = Exp(x)
	assert x1.val == [np.exp(2)]
	assert x1.der['x'] == [np.exp(2)]

def test_exp2():
	exp = Exp(5)
	assert exp.val == [np.exp(5)]
	assert exp.der['x'] == [0]

def test_create_abs():
	x = X(2)
	a = Abs(x)
	assert a.val == [2]
	assert a.der['x'] == [1]

def test_abs2():
	a = Abs(-3)
	assert a.val == [3]
	assert a.der['x'] == [0]

def test_neg():
	x = X(2)
	x1 = -x
	assert x1.val == [-2]
	assert x1.der['x'] == [-1]

def test_add():
	x = X(2)
	f = x + 3

	assert f.val == [5]
	assert f.der['x'] == [1]


def test_add1():
	x = X(2)
	f = x + x
	assert f.val == [4]
	assert f.der['x'] == [2]

def test_add2():
	x = X(2, 'x')
	y = X(3, 'y')
	f = (5 * x) + (8 * y)
	assert f.val == [34]
	assert f.der['x'] == [5]
	assert f.der['y'] == [8]


def test_radd():
	x = X(2)
	f = 3 + x
	assert f.val == [5]
	assert f.der['x'] == [1]


def test_sub():
	x = X(2)
	f = x - 3
	assert f.val == [-1]
	assert f.der['x'] == [1]


def test_sub1():
	x = X(2)
	f = x - x
	assert f.val == [0]
	assert f.der['x'] == [0]


def test_rsub():
	x = X(2)
	f = 3 - x
	assert f.val == [1]
	assert f.der['x'] == [-1]


def test_mul():
	x = X(2)
	f = x * 3
	assert f.val == [6]
	assert f.der['x'] == [3]


def test_mul1():
	x = X(2)
	f = x * x
	assert f.val == [4]
	assert f.der['x'] == [4]

def test_mul2():
	x = X(4, 'x')
	y = X(5, 'y')
	f = (2 * x) * (3 * y)
	assert f.val == [120]
	assert f.der['x'] == [30]
	assert f.der['y'] == [24]


def test_rmul():
	x = X(2)
	f = 3 * x
	assert f.val == [6]
	assert f.der['x'] == [3]


def test_div():
	x = X(2)
	f = x / 2
	assert f.val == [1]
	assert f.der['x'] == [0.5]


def test_div1():
	x = X(2)
	f = x / x
	assert f.val == [1]
	assert f.der['x'] == [0]


def test_rdiv():
	x = X(2)
	f = 2 / x
	assert f.val == [1]
	assert f.der['x'] == [-0.5]


def test_pow():
	x = X(2)
	f = x ** 3
	assert f.val == [8]
	assert f.der['x'] == [12]

def test_pow2():
	x = X(2)
	f = x ** x
	assert f.val == [4]
	assert f.der['x'] == [4 + np.log(16)]

def test_pow3():
	x = X(2, 'x')
	y = X(2, 'y')
	f = (5 * x) ** (2 * y)
	assert f.val == [10000]
	assert f.der['x'] == [20000]
	assert f.der['y'] == [20000 * np.log(10)]

def test_rpow():
	x = X(2)
	f = 3 ** x
	assert f.val == [9]
	assert f.der['x'] == [9 * np.log(3)]

def test_arcsin():
	x = X(0.5)
	f = Arcsin(x)
	assert f.val == [np.arcsin(0.5)]
	assert f.der['x'] == [1 / np.sqrt(1 - 0.5 ** 2)]

def test_arcsin2():
	f = Arcsin(0.5)
	assert f.val == [np.arcsin(0.5)]
	assert f.der['x'] == [0]

def test_arccos():
	x = X(0.5)
	f = Arccos(x)
	assert f.val == [np.arccos(0.5)]
	assert f.der['x'] == [-1 / np.sqrt(1 - 0.5 ** 2)]

def test_arccos2():
	f = Arccos(0.5)
	assert f.val == [np.arccos(0.5)]
	assert f.der['x'] == [0]

def test_arctan():
	x = X(0.5)
	f = Arctan(x)
	assert f.val == [np.arctan(0.5)]
	assert f.der['x'] == [1 / ((0.5 ** 2) + 1)]

def test_arctan2():
	f = Arctan(0.5)
	assert f.val == [np.arctan(0.5)]
	assert f.der['x'] == [0]

def test_sinh():
	x = X(2)
	f = Sinh(x)
	assert f.val == [pytest.approx(np.sinh(2))]
	assert f.der['x'] == [pytest.approx(np.cosh(2))]

def test_sinh2():
	f = Sinh(2)
	assert f.val == [pytest.approx(np.sinh(2))]
	assert f.der['x'] == [0]

def test_cosh():
	x = X(2)
	f = Cosh(x)
	assert f.val == [pytest.approx(np.cosh(2))]
	assert f.der['x'] == [pytest.approx(np.sinh(2))]

def test_cosh2():
	f = Cosh(2)
	assert f.val == [pytest.approx(np.cosh(2))]
	assert f.der['x'] == [0]

def test_tanh():
	x = X(2)
	f = Tanh(x)
	assert f.val == [pytest.approx(np.tanh(2))]
	assert f.der['x'] == [pytest.approx(1 / (np.cosh(2) ** 2))]

def test_logistic():
	x = X(2)
	f = Logistic(x)
	assert f.val == [1 / (1 + np.exp(-2))]
	assert f.der['x'] == [np.exp(-2) / ((1 + np.exp(-2)) ** 2)]

def test_sqrt():
	x = X(2)
	f = Sqrt(x)
	assert f.val == [np.sqrt(2)]
	assert f.der['x'] == [0.5 * 2 ** (-0.5)]

def test_variables():
	x = X(2, 'x')
	y = X(3, 'y')
	f = x * y
	assert f.variables == {'x', 'y'}