import pytest
import numpy as np

from superdifferentiator.forward.functions import X, Sin, Cos, Tan, Ln, Log, Exp, Abs

# Error message for passing invalid argument to an elementary function
E1 = 'Invalid first argument, must be an AD object or a number.'

def test_create_x():
	x = X(2)
	assert x.val == 2
	assert x.der['x'] == 1


def test_create_cos():
	x = X(2)
	x1 = Cos(x)
	assert x1.val == np.cos(2)
	assert x1.der['x'] == -np.sin(2)

def test_cos2():
	c = Cos(8)
	assert c.val == np.cos(8)
	assert c.der['x'] == 0


def test_create_sin():
	x = X(2)
	x1 = Sin(x)
	assert x1.val == np.sin(2)
	assert x1.der['x'] == np.cos(2)

def test_sin2():
	s = Sin(5)
	assert s.val == np.sin(5)
	assert s.der['x'] == 0

def test_create_tan():
	x = X(2)
	x1 = Tan(x)
	assert x1.val == np.tan(2)
	assert x1.der['x'] == 1 / np.cos(2) ** 2

def test_tan2():
	t = Tan(11)
	assert t.val == np.tan(11)
	assert t.der['x'] == 0

def test_create_ln():
	x = X(2)
	x1 = Ln(x)
	assert x1.val == np.log(2)
	assert x1.der['x'] == 0.5

def test_ln2():
	ln = Ln(5)
	assert ln.val ==  np.log(5)
	assert ln.der['x'] == 0

def test_create_log():
	x = X(2)
	log = Log(x, 10)
	assert log.val == np.log(2) / np.log(10)
	assert log.der['x'] == 1 / (2 * np.log(10))

def test_log2():
	log = Log(100, 10)
	assert log.val == 2
	assert log.der['x'] == 0

def test_create_exp():
	x = X(2)
	x1 = Exp(x)
	assert x1.val == np.exp(2)
	assert x1.der['x'] == np.exp(2)

def test_exp2():
	exp = Exp(5)
	assert exp.val == np.exp(5)
	assert exp.der['x'] == 0

def test_create_abs():
	x = X(2)
	a = Abs(x)
	assert a.val == 2
	assert a.der['x'] == 1

def test_abs2():
	a = Abs(-3)
	assert a.val == 3
	assert a.der['x'] == 0

def test_neg():
	x = X(2)
	x1 = -x
	assert x1.val == -2
	assert x1.der['x'] == -1

def test_add():
	x = X(2)
	f = x + 3

	assert f.val == 5
	assert f.der['x'] == 1


def test_add1():
	x = X(2)
	f = x + x
	assert f.val == 4
	assert f.der['x'] == 2

def test_add2():
	x = X(2, 'x')
	y = X(3, 'y')
	f = (5 * x) + (8 * y)
	assert f.val == 34
	assert f.der['x'] == 5
	assert f.der['y'] == 8


def test_radd():
	x = X(2)
	f = 3 + x
	assert f.val == 5
	assert f.der['x'] == 1


def test_sub():
	x = X(2)
	f = x - 3
	assert f.val == -1
	assert f.der['x'] == 1


def test_sub1():
	x = X(2)
	f = x - x
	assert f.val == 0
	assert f.der['x'] == 0


def test_rsub():
	x = X(2)
	f = 3 - x
	assert f.val == 1
	assert f.der['x'] == -1


def test_mul():
	x = X(2)
	f = x * 3
	assert f.val == 6
	assert f.der['x'] == 3


def test_mul1():
	x = X(2)
	f = x * x
	assert f.val == 4
	assert f.der['x'] == 4

def test_mul2():
	x = X(4, 'x')
	y = X(5, 'y')
	f = (2 * x) * (3 * y)
	assert f.val == 120
	assert f.der['x'] == 30
	assert f.der['y'] == 24


def test_rmul():
	x = X(2)
	f = 3 * x
	assert f.val == 6
	assert f.der['x'] == 3


def test_div():
	x = X(2)
	f = x / 2
	assert f.val == 1
	assert f.der['x'] == 0.5


def test_div1():
	x = X(2)
	f = x / x
	assert f.val == 1
	assert f.der['x'] == 0


def test_rdiv():
	x = X(2)
	f = 2 / x
	assert f.val == 1
	assert f.der['x'] == -0.5


def test_pow():
	x = X(2)
	f = x ** 3
	assert f.val == 8
	assert f.der['x'] == 12

def test_pow2():
	x = X(2)
	f = x ** x
	assert f.val == 4
	assert f.der['x'] == 4 + np.log(16)

def test_pow3():
	x = X(2, 'x')
	y = X(2, 'y')
	f = (5 * x) ** (2 * y)
	assert f.val == 10000
	assert f.der['x'] == 20000
	assert f.der['y'] == 20000 * np.log(10)

def test_rpow():
	x = X(2)
	f = 3 ** x
	assert f.val == 9
	assert f.der['x'] == 9 * np.log(3)


def add_invalid():
	x = X(2)
	f = x + 'hello'


def test_invalid_add():
	with pytest.raises(ValueError) as excinfo:
		add_invalid()
	assert 'Operand in addition is invalid. Operand must be an AD object or a number.' == str(excinfo.value)


def mul_invalid():
	x = X(2)
	f = x * 'hello'


def test_invalid_mul():
	with pytest.raises(ValueError) as excinfo:
		mul_invalid()
	assert 'Operand in multiplication is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def sub_invalid():
	x = X(2)
	f = x - 'hello'

def test_invalid_sub():
	with pytest.raises(ValueError) as excinfo:
		sub_invalid()
	assert 'Operand in subtraction is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def rsub_invalid():
	x = X(2)
	f = 'hello' - x

def test_invalid_rsub():
	with pytest.raises(ValueError) as excinfo:
		rsub_invalid()
	assert 'Operand in subtraction is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def div_invalid():
	x = X(2)
	f = x / 'hello'

def test_invalid_dif():
	with pytest.raises(ValueError) as excinfo:
		div_invalid()
	assert 'Operand in division is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def rdiv_invalid():
	x = X(2)
	f = 'hello' / x

def test_invalid_rdiv():
	with pytest.raises(ValueError) as excinfo:
		rdiv_invalid()
	assert 'Operand in division is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def pow_invalid():
	x = X(2)
	f = x ** 'hello'

def test_invalid_pow():
	with pytest.raises(ValueError) as excinfo:
		pow_invalid()
	assert 'Operand in power is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def rpow_invalid():
	x = X(2)
	f = 'hello' ** x

def test_invalid_rpow():
	with pytest.raises(ValueError) as excinfo:
		rpow_invalid()
	assert 'Operand in power is invalid. Operand must be an AD object or a number.' == str(excinfo.value)

def ln_invalid():
	ln = Ln('Hello')

def test_invalid_ln():
	with pytest.raises(ValueError) as excinfo:
		ln_invalid()
	assert E1 == str(excinfo.value)

def log_invalid():
	log = Log('Hello', 5)

def test_invalid_log():
	with pytest.raises(ValueError) as excinfo:
		log_invalid()
	assert E1 == str(excinfo.value)

def exp_invalid():
	exp = Exp('Hello')

def test_invalid_exp():
	with pytest.raises(ValueError) as excinfo:
		exp_invalid()
	assert E1 == str(excinfo.value)

def abs_invalid():
	a = Abs('Hello')

def test_invalid_abs():
	with pytest.raises(ValueError) as excinfo:
		abs_invalid()
	assert E1 == str(excinfo.value)

def sin_invalid():
	s = Sin('asdf')

def test_invalid_sin():
	with pytest.raises(ValueError) as excinfo:
		sin_invalid()
	assert E1 == str(excinfo.value)

def cos_invalid():
	c = Cos('asdf')

def test_invalid_cos():
	with pytest.raises(ValueError) as excinfo:
		cos_invalid()
	assert E1 == str(excinfo.value)

def tan_invalid():
	t = Tan('weqr')

def test_invalid_tan():
	with pytest.raises(ValueError) as excinfo:
		tan_invalid()
	assert E1 == str(excinfo.value)





















