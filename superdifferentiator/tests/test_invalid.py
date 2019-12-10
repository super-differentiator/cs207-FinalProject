import pytest

from superdifferentiator.forward.functions import X, Sin, Cos, Tan, Ln, Log, Exp, Abs, Sqrt, Arcsin, Arccos, Arctan, Sinh, Cosh, Tanh, Logistic

# Error message for passing invalid argument to an elementary function
E1 = 'Invalid first argument, must be an AD object or a number.'

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

def arcsin_invalid():
	t = Arcsin('wieojf')

def test_invalid_arcsin():
	with pytest.raises(ValueError) as excinfo:
		arcsin_invalid()
	assert E1 == str(excinfo.value)

def arccos_invalid():
	t = Arccos('wfejio')

def test_invalid_arccos():
	with pytest.raises(ValueError) as excinfo:
		arccos_invalid()
	assert E1 == str(excinfo.value)

def arctan_invalid():
	t = Arctan('jwe')

def test_invalid_arctan():
	with pytest.raises(ValueError) as excinfo:
		arctan_invalid()
	assert E1 == str(excinfo.value)