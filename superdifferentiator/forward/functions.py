# CS207 Final Project

import numpy as np
from superdifferentiator.forward.AD import AD

class X(AD):
	'''The X class is the base function class that you begin with, representing
	the function f(x) = x. This class extends from the AD class to store the function
	value and derivative. You use this to build larger functions involving constants,
	exponents, and elementary functions like sin, cos, log, etc.
	'''

	def __init__(self, alpha, name = 'x'):
		# Dictionary of partial derivatives. Currently only one var, so only one partial
		ders = {name: 1}

		super().__init__(alpha, ders)

class Ln(AD):
	'''The Ln class implements the natural log function. You initialize an Ln object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		if isinstance(fun, AD):
			# Natural log of a function, ln[f(x)]
			val = np.log(fun.val)

			der = {}
			for var in fun.der.keys():
				der[var] = fun.der[var] / fun.val

		elif isinstance(fun, (int, float)):
			# Natural log of a constant, ln(c)
			val = np.log(fun)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

class Log(AD):
	'''The Log class implements the general log function with a given base. You initialize
	a Log object by passing in another function object or a constant, and the base of the
	log. The function value and derivative will be calculated.
	'''

	def __init__(self, fun, a):
		val = 0
		der = {}

		if isinstance(fun, AD):
			# Log base a of function, log_a[f(x)]
			val = np.log(fun.val) / np.log(a)

			der = {}
			for var in fun.der.keys():
				der[var] = fun.der[var] / (np.log(a) * fun.val)

		elif isinstance(fun, (int, float)):
			# Log base a of a constant, log_a(c)
			val = np.log(fun) / np.log(a)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

class Exp(AD):
	'''The Exp class implements the exponential function, exp[f(x)]. You initialize
	an Exp object by passing in another function object or a constant, and the function
	value and derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		if isinstance(fun, AD):
			# e raised to a function, exp[f(x)]
			val = np.exp(fun.val)

			der = {}
			for var in fun.der.keys():
				der[var] = np.exp(fun.val) * fun.der[var]

		elif isinstance(fun, (int, float)):
			# e raised to a constant, exp(c)
			val = np.exp(fun)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

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

		if isinstance(fun, AD):
			# Absolute value of a function, |f(x)|
			val = np.abs(fun.val)

			der = {}
			for var in fun.der.keys():
				der[var] = fun.val * fun.der[var] / np.abs(fun.val)

		elif isinstance(fun, (int, float)):
			# Absolute value of a constant, |c|
			val = np.abs(fun)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

class Sin(AD):
	'''The Sin class implements the sin function. You initialize a Sin object by
	passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		if isinstance(fun, AD):
			# Sin of a function, sin[f(x)]
			val = np.sin(fun.val)

			der = {}
			for var in fun.der.keys():
				der[var] = fun.der[var] * np.cos(fun.val)

		elif isinstance(fun, (int, float)):
			# Sin of a constant, sin(c)
			val = np.sin(fun)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

class Cos(AD):
	'''The Cos class implements the cosine function. You initialize a Cos object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		if isinstance(fun, AD):
			# Cosine of a function, cos[f(x)]
			val = np.cos(fun.val)

			for var in fun.der.keys():
				der[var] = -fun.der[var] * np.sin(fun.val)

		elif isinstance(fun, (int, float)):
			# Cosine of a constant, cos(c)
			val = np.cos(fun)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

class Tan(AD):
	'''The Tan class implements the tangent function. You initialize a Tan object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		val = 0
		der = {}

		if isinstance(fun, AD):
			# Tangent of a function, tan[f(x)]
			val = np.tan(fun.val)

			der = {}
			for var in fun.der.keys():
				der[var] = fun.der[var] / (np.cos(fun.val) ** 2)

		elif isinstance(fun, (int, float)):
			val = np.tan(fun)
			der['x'] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)
