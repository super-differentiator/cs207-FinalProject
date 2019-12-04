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
		der = 0

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
			val = self.val + other
			der = self.der.copy()
		else:
			raise ValueError('Operand in addition is invalid. Operand must be an AD object or a number.')

		return AD(val, der)

	def __radd__(self, other):
		return self + other

	def __mul__(self, other):
		val = 0
		der = 0

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
		try:
			return self + (-1) * other
		except ValueError:
			raise ValueError('Operand in subtraction is invalid. Operand must be an AD object or a number.')

	def __rsub__(self, other):
		try:
			return (-1) * self + other
		except ValueError:
			raise ValueError('Operand in subtraction is invalid. Operand must be an AD object or a number.')

	def __neg__(self):
		return (-1) * self

	def __truediv__(self, other):
		try:
			return self * (other ** -1)
		except TypeError:
			raise ValueError('Operand in division is invalid. Operand must be an AD object or a number.')

	def __rtruediv__(self, other):
		try:
			return (self ** -1) * other
		except ValueError:
			raise ValueError('Operand in division is invalid. Operand must be an AD object or a number.')

	def __pow__(self, other):
		val = 0
		der = 0

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
		der = 0

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