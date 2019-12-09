import pytest
import numpy as np

from superdifferentiator.forward.functions import X
from superdifferentiator.forward.Vector import Vector

def test_vector1():
	xV, yV = 2, 3
	x, y = X(xV, 'x'), X(yV, 'y')
	f1, f2 = 2 * (x ** 2) * y, 3 * (x ** 2) + (y ** 3)

	v = Vector([f1, f2])
	assert v.val[0][0, 0] == 2 * (xV ** 2) * yV
	assert v.val[0][1, 0] == 3 * (xV ** 2) + (yV ** 3)

	allVars, jac = v.jacobian()
	jac = jac[0]
	xIdx = allVars.index('x')
	yIdx = allVars.index('y')
	assert jac[0, xIdx] == 4 * xV * yV
	assert jac[0, yIdx] == 2 * (xV ** 2)
	assert jac[1, xIdx] == 6 * xV
	assert jac[1, yIdx] == 3 * (yV ** 2)

def test_vector2():
	xV, yV = 2, 3
	x, y = X(xV, 'x'), X(yV, 'y')
	f1, f2 = (x ** 2) + (2 * x) + 3, 3 * (y ** 2) + (4 * y) + 8

	v = Vector([f1, f2])
	assert v.val[0][0, 0] == (xV ** 2) + (2 * xV) + 3
	assert v.val[0][1, 0] == 3 * (yV ** 2) + (4 * yV) + 8

	allVars, jac = v.jacobian()
	jac = jac[0]
	xIdx = allVars.index('x')
	yIdx = allVars.index('y')
	assert jac[0, xIdx] ==  2 * xV + 2
	assert jac[0, yIdx] == 0
	assert jac[1, xIdx] == 0
	assert jac[1, yIdx] == 6 * yV + 4

def test_vector_operators():
	x1 = X(3)
	v1 = Vector([x1])
	x2 = X(4)
	v2 = Vector([x2])

	assert v1 < v2
	assert v1 <= v2
	assert not v1 > v2
	assert not v1 >= v2

	assert v1 > 2
	assert v1 >= 2
	assert v1 < 4
	assert v1 <= 4

	assert v1 == v1
	assert v1 != v2

	x3 = X(5, 'x3')
	v3 = Vector([x1, x3])
	assert v1 != v3
