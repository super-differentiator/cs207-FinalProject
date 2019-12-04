# CS207 Final Project

import numpy as np

class AD:
	'''The AD class stores the function value and the derivative for a function f(x),
	as well as the derivative rules for addition, subtraction, multiplication, division,
	and raising to a power, by overloading the +, -, *, /, and ** operators.
	'''

	def __init__(self, val, der):
		self._val = val
		self._der = der # Dictionary of partial derivatives

	@property
	def val(self):
		return self._val

	@property
	def der(self):
		return self._der

	def __add__(self, other):
		val = 0
		der = {}

		if isinstance(other, AD):
			# Adding two functions, f + g
			val = self.val + other.val

			# Get list of all variables of the two functions
			myVars, otherVars, allVars = getVars(self.der, other.der)

			# Dictionary of new partial derivatives
			der = {}

			# Compute each partial derivative
			for var in allVars:
				if var in myVars and var in otherVars:
					der[var] = self.der[var] + other.der[var]
				elif var in myVars and var not in otherVars:
					der[var] = self.der[var]
				else:
					der[var] = other.der[var]

		elif isinstance(other, (int, float)):
			# Adding a function and scalar, f(x) + c
			v = self.val + other
			der = self.der.copy()
		else:
			raise ValueError('Operand in addition is invalid. Operand must be an AD object or a number.')

		return AD(val, der)

	def __radd__(self, other):
		return self + other

	def __mul__(self, other):
		val = 0
		der = {}

		if isinstance(other, AD):
			# Multiplying two functions, f(x) * g(x)
			val = self.val * other.val

			# Get list of all variables of the two functions
			myVars, otherVars, allVars = getVars(self.der, other.der)

			# Dictionary of new partial derivatives
			der = {}

			for var in allVars:
				if var in myVars and var in otherVars:
					der[var] = self.der[var] * other.val + self.val * other.der[var]
				elif var in myVars and var not in otherVars:
					der[var] = other.val * self.der[var]
				else:
					der[var] = self.val * other.der[var]

		elif isinstance(other, (int, float)):
			# Multiplying the function by a constant, f(x) * c
			val = self.val * other

			# Multiply each partial by the constant
			der = {}
			for key in self.der.keys():
				der[key] = self.der[key] * other

		else:
			raise ValueError('Operand in multiplication is invalid. Operand must be an AD object or a number.')

		return AD(val, der)

	def __rmul__(self, other):
		return self * other

	def __sub__(self, other):
		return self + (-1) * other

	def __rsub__(self, other):
		return (-1) * self + other

	def __neg__(self):
		return (-1) * self

	def __truediv__(self, other):
		return self * (other ** -1)

	def __rtruediv__(self, other):
		return (self ** -1) * other

	def __pow__(self, other):
		val = 0
		der = {}

		if isinstance(other, AD):
			# Raising function to a function, f(x) ^ g(x)
			val = self.val ** other.val

			# Get list of all variables of the two functions
			myVars, otherVars, allVars = getVars(self.der, other.der)

			# Dictionary of new partial derivatives
			der = {}

			for var in allVars:
				if var in myVars and var in otherVars: # f(x, y) ^ g(x, y)
					der[var] = (self.val ** (other.val - 1)) * (self.der[var] * other.val + self.val * other.der[var] * np.log(self.val))
				elif var in myVars and var not in otherVars: # f(x) ^ g(y)
					der[var] = other.val * self.der[var] * (self.val ** (other.val - 1))
				else: # f(y) ^ g(x)
					der[var] = np.log(self.val) * other.der[var] * (self.val ** other.val)

			#d = self.val ** (other.val - 1) * (other.val * self.der + self.val * np.log(self.val) * other.der)
		elif isinstance(other, (int, float)):
			# Raising the function to a constant, f(x) ^ c
			val = self.val ** other

			der = {}

			for var in self.der.keys():
				der[var] = other * self.der[var] * (self.val ** (other - 1))

		else:
			raise ValueError('Operand in power is invalid. Operand must be an AD object or a number.')

		return AD(val, der)

	def __rpow__(self, other):
		val = 0
		der = {}

		if isinstance(other, (int, float)):
			# Raising constant to the function, c ^ f(x)
			val = other ** self.val

			der = {}

			for var in self.der.keys():
				der[var] = np.log(other) * self.der[var] * (other ** self.val)

		else:
			raise ValueError('Operand in power is invalid. Operand must be an AD object or a number.')

		return AD(val, der)

def getVars(der1, der2):
	# Get list of all variables of the two functions
	vars1 = set(der1.keys())
	vars2 = set(der2.keys())
	allVars = vars1.union(vars2)

	return vars1, vars2, allVars

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

			for var in fun.der.keys():
				der[var] = 0
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

			for var in fun.der.keys():
				der[var] = 0
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

			for var in fun.der.keys():
				der[var] = 0
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

			for var in fun.der.keys():
				der[var] = 0
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

			for var in fun.der.keys():
				der[var] = 0
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

			der = {}
			for var in fun.der.keys():
				der[var] = -fun.der[var] * np.sin(fun.val)

		elif isinstance(fun, (int, float)):
			# Cosine of a constant, cos(c)
			val = np.cos(fun)

			for var in fun.der.keys():
				der[var] = 0
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

			for var in fun.der.keys():
				der[var] = 0
		else:
			raise ValueError('Invalid first argument, must be an AD object or a number.')

		super().__init__(val, der)

def main():
	x = X(3, 'x')
	y = X(4, 'y')

	s = Sin(Ln(x * y))
	print(s.val, s.der)

if __name__ == '__main__':
	main()