# CS207 Final Project

import numpy as np
from AutoDifferentiation.forward.AD import AD

class X(AD):
	'''The X class is the base function class that you begin with, representing
	the function f(x) = x. This class extends from the AD class to store the function
	value and derivative. You use this to build larger functions involving constants,
	exponents, and elementary functions like sin, cos, log, etc.
	'''

	def __init__(self, alpha):
		super().__init__(alpha, 1)

class Ln(AD):
	'''The Ln class implements the natural log function. You initialize an Ln object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# Natural log of a function, ln[f(x)]
			v = np.log(fun.val)
			d = fun.der / fun.val
		elif isinstance(fun, (int, float)):
			# Natural log of a constant, ln(c)
			v = np.log(fun)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)

class Log(AD):
	'''The Log class implements the general log function with a given base. You initialize
	a Log object by passing in another function object or a constant, and the base of the
	log. The function value and derivative will be calculated.
	'''

	def __init__(self, fun, a):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# Log base a of function, log_a[f(x)]
			v = np.log(fun.val) / np.log(a)
			d = fun.der / (np.log(a) * fun.val)
		elif isinstance(fun, (int, float)):
			# Log base a of a constant, log_a(c)
			v = np.log(fun) / np.log(a)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)

class Exp(AD):
	'''The Exp class implements the exponential function, exp[f(x)]. You initialize
	an Exp object by passing in another function object or a constant, and the function
	value and derivative will be calculated.
	'''

	def __init__(self, fun):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# e raised to a function, exp[f(x)]
			v = np.exp(fun.val)
			d = fun.der * np.exp(fun.val)
		elif isinstance(fun, (int, float)):
			# e raised to a constant, exp(c)
			v = np.exp(fun)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)

class Abs(AD):
	'''The Abs class implements the absolute value function, |f(x)|. You initialize
	an Abs object by passing in another function object or a constant, and the function
	value and derivative will be calculated.

	WARNING: if the derivative is evaluated at the cusp, a warning will be issued and
	the derivative value will be nan.
	'''

	def __init__(self, fun):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# Absolute value of a function, |f(x)|
			v = np.abs(fun.val)
			d = fun.val * fun.der / np.abs(fun.val)
		elif isinstance(fun, (int, float)):
			# Absolute value of a constant, |c|
			v = np.abs(fun)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)

class Sin(AD):
	'''The Sin class implements the sin function. You initialize a Sin object by
	passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# Sin of a function, sin[f(x)]
			v = np.sin(fun.val)
			d = fun.der * np.cos(fun.val)
		elif isinstance(fun, (int, float)):
			# Sin of a constant, sin(c)
			v = np.sin(fun)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)

class Cos(AD):
	'''The Cos class implements the cosine function. You initialize a Cos object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# Cosine of a function, cos[f(x)]
			v = np.cos(fun.val)
			d = -fun.der * np.sin(fun.val)
		elif isinstance(fun, (int, float)):
			# Cosine of a constant, cos(c)
			v = np.cos(fun)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)

class Tan(AD):
	'''The Tan class implements the tangent function. You initialize a Tan object
	by passing in another function object or a constant, and the function value and
	derivative will be calculated.
	'''

	def __init__(self, fun):
		v = 0
		d = 0

		if isinstance(fun, AD):
			# Tangent of a function, tan[f(x)]
			v = np.tan(fun.val)
			d = fun.der / (np.cos(fun.val) ** 2)
		elif isinstance(fun, (int, float)):
			v = np.tan(fun)
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(v, d)