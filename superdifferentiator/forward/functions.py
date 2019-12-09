# CS207 Final Project

import numpy as np
from superdifferentiator.forward.AD import AD

class X(AD):
	'''The X class is the base function class that you begin with, representing
	the function f(x) = x. This class extends from the AD class to store the function
	value and derivative. You use this to build larger functions involving constants,
	exponents, and elementary functions like sin, cos, log, etc.
	'''

	def __init__(self, alphas, name = 'x'):
		# If a scalar is given, convert it to a list
		if isinstance(alphas, (int, float)):
			alphas = [alphas]

		# Dictionary of partial derivatives. Currently only one var, so only one partial
		ders = {name: [1] * len(alphas)}

		s = 'X(' + name + ', \'' + name + '\')'
		repr_s = 'X(' + str(alphas) + ', \'' + name + '\')'

		super().__init__(alphas, ders, s, repr_s)

class Ln(AD):
	'''The Ln class implements the natural log function. You initialize an Ln object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Ln(' + str(fun) + ')'
		repr_s = 'Ln(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)
			der = {}
			for v in fun.der.keys():
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				val[i1] = np.log(fun.val[i1])

				for var in fun.der.keys():
					der[var][i1] = fun.der[var][i1] / fun.val[i1]
		elif isinstance(fun, (int, float)):
			val = [np.log(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')
			
		super().__init__(val, der, s, repr_s)

class Log(AD):
	'''The Log class implements the general log function with a given base. You initialize
	a Log object by passing in another function object or a constant, and the base of the
	log. The function value and derivative will be calculated.
	'''

	def __init__(self, fun, a):
		t = Ln(fun) / np.log(a)
		super().__init__(t.val, t.der, str(t), repr(t))

class Exp(AD):
	'''The Exp class implements the exponential function, exp[f(x)]. You initialize
	an Exp object by passing in another function object or a constant, and the function
	value and derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Exp(' + str(fun) + ')'
		repr_s = 'Exp(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0]

			for v in fun.der.keys():
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				# e raised to a function, exp[f(x)]
				val[i1] = np.exp(fun.val[i1])

				for var in fun.der.keys():
					der[var][i1] = np.exp(fun.val[i1]) * fun.der[var][i1]

		elif isinstance(fun, (int, float)):
			# e raised to a constant, exp(c)
			val = [np.exp(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Abs(AD):
	'''The Abs class implements the absolute value function, |f(x)|. You initialize
	an Abs object by passing in another function object or a constant, and the function
	value and derivative will be calculated.

	WARNING: if the derivative is evaluated at the cusp, a warning will be issued and
	the derivative value will be nan.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Abs(' + str(fun) + ')'
		repr_s = 'Abs(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.der.keys():
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				val[i1] = np.abs(fun.val[i1])

				for var in fun.der.keys():
					der[var][i1] = fun.val[i1] * fun.der[var][i1] / np.abs(fun.val[i1])

		elif isinstance(fun, (int, float)):
			# Absolute value of a constant, |c|
			val = [np.abs(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Sin(AD):
	'''The Sin class implements the sin function. You initialize a Sin object by
	passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Sin(' + str(fun) + ')'
		repr_s = 'Sin(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.der.keys():
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				# Sin of a function, sin[f(x)]
				val[i1] = np.sin(fun.val[i1])

				for var in fun.der.keys():
					der[var][i1] = fun.der[var][i1] * np.cos(fun.val[i1])

		elif isinstance(fun, (int, float)):
			# Sin of a constant, sin(c)
			val = [np.sin(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Cos(AD):
	'''The Cos class implements the cosine function. You initialize a Cos object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Cos(' + str(fun) + ')'
		repr_s = 'Cos(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.der.keys():
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				# Cosine of a function, cos[f(x)]
				val[i1] = np.cos(fun.val[i1])

				for var in fun.der.keys():
					der[var][i1] = -fun.der[var][i1] * np.sin(fun.val[i1])

		elif isinstance(fun, (int, float)):
			# Cosine of a constant, cos(c)
			val = [np.cos(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Tan(AD):
	'''The Tan class implements the tangent function. You initialize a Tan object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Tan(' + str(fun) + ')'
		repr_s = 'Tan(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.der.keys():
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				# Tangent of a function, tan[f(x)]
				val[i1] = np.tan(fun.val[i1])

				for var in fun.der.keys():
					der[var][i1] = fun.der[var][i1] / (np.cos(fun.val[i1]) ** 2)

		elif isinstance(fun, (int, float)):
			val = [np.tan(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Sqrt(AD):
	def __init__(self, fun):
		t = fun ** 0.5
		super().__init__(t.val, t.der, str(t), repr(t))

class Arcsin(AD):
	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Arcsin(' + str(fun) + ')'
		repr_s = 'Arcsin(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.variables:
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				val[i1] = np.arcsin(fun.val[i1])

				for var in fun.variables:
					der[var][i1] = fun.der[var][i1] / np.sqrt(1 - fun.val[i1] ** 2)

		elif isinstance(fun, (int, float)):
			val = [np.arcsin(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Arccos(AD):
	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Arccos(' + str(fun) + ')'
		repr_s = 'Arccos(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.variables:
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				val[i1] = np.arccos(fun.val[i1])

				for var in fun.variables:
					der[var][i1] = -fun.der[var][i1] / np.sqrt(1 - fun.val[i1] ** 2)

		elif isinstance(fun, (int, float)):
			val = [np.arccos(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Arctan(AD):
	def __init__(self, fun):
		val = 0
		der = {}

		s = 'Arctan(' + str(fun) + ')'
		repr_s = 'Arctan(' + repr(fun) + ')'

		if isinstance(fun, AD):
			val = [0] * len(fun.val)

			for v in fun.variables:
				der[v] = [0] * len(fun.val)

			for i1 in range(len(fun.val)):
				val[i1] = np.arctan(fun.val[i1])

				for var in fun.variables:
					der[var][i1] = fun.der[var][i1] / ((fun.val[i1] ** 2) + 1)
		elif isinstance(fun, (int, float)):
			val = [np.arctan(fun)]
			der['x'] = [0]
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der, s, repr_s)

class Sinh(AD):
	def __init__(self, fun):
		t = (Exp(fun) - Exp(-fun)) / 2
		super().__init__(t.val, t.der, str(t), repr(t))

class Cosh(AD):
	def __init__(self, fun):
		t = (Exp(fun) + Exp(-fun)) / 2
		super().__init__(t.val, t.der, str(t), repr(t))

class Tanh(AD):
	def __init__(self, fun):
		t = Sinh(fun) / Cosh(fun)
		super().__init__(t.val, t.der, str(t), repr(t))

class Logistic(AD):
	def __init__(self, fun):
		t = 1 / (1 + Exp(-fun))
		super().__init__(t.val, t.der, str(t), repr(t))
