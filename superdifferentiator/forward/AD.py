# CS207 Final Project

import numpy as np

class AD:
	'''The AD class stores the function value and the derivative for a function f(x),
	as well as the derivative rules for addition, subtraction, multiplication, division,
	and raising to a power, by overloading the +, -, *, /, and ** operators.
	'''

	def __init__(self, vals, der, s, repr_s):
		self._vals = vals
		self._der = der # Dictionary of partial derivatives
		self._s = s
		self._repr_s = repr_s

	@property
	def val(self):
		return self._vals

	@property
	def der(self):
		return self._der

	@property
	def variables(self):
		return set(self.der.keys())

	@property
	def num_vals(self):
		return len(self.val)

	def __str__(self):
		return self._s

	def __repr__(self):
		return self._repr_s

	def jacobian(self):
		allVars = list(self.variables)
		jacs = []
		for i1 in range(self.num_vals):
			jacs.append(np.zeros((1, len(allVars))))

		for i1 in range(self.num_vals):
			for col in range(len(allVars)):
				jacs[i1][0, col] = self.der[allVars[col]][i1]

		return allVars, jacs

	def __add__(self, other):
		val = [0] * self.num_vals
		der = {}
		s = '(' + str(self) + ') + (' + str(other) + ')'
		repr_s = '(' + repr(self) + ') + (' + repr(other) + ')'

		if isinstance(other, AD):
			myVars, otherVars, allVars = getVars(self.der, other.der)
			for v in allVars:
				der[v] = [0] * self.num_vals

			for i1 in range(self.num_vals):
				# Adding two functions, f + g
				val[i1] = self.val[i1] + other.val[i1]

				# Compute each partial derivative
				for var in allVars:
					if var in myVars and var in otherVars:
						der[var][i1] = self.der[var][i1] + other.der[var][i1]
					elif var in myVars and var not in otherVars:
						der[var][i1] = self.der[var][i1]
					else:
						der[var][i1] = other.der[var][i1]

		elif isinstance(other, (int, float)):
			val = [v + other for v in self.val]
			der = self.der.copy()
		else:
			raise ValueError('Operand in addition is invalid. Operand must be an AD object or a number.')

		return AD(val, der, s, repr_s)

	def __radd__(self, other):
		return self + other

	def __mul__(self, other):
		val = 0
		der = {}
		s = '(' + str(self) + ') * (' + str(other) + ')'
		repr_s = '(' + repr(self) + ') * (' + repr(other) + ')'

		if isinstance(other, AD):
			val = [0] * self.num_vals

			myVars, otherVars, allVars = getVars(self.der, other.der)
			for v in allVars:
				der[v] = [0] * self.num_vals

			for i1 in range(self.num_vals):
				# Multiplying two functions, f(x) * g(x)
				val[i1] = self.val[i1] * other.val[i1]

				for var in allVars:
					if var in myVars and var in otherVars:
						der[var][i1] = self.der[var][i1] * other.val[i1] + self.val[i1] * other.der[var][i1]
					elif var in myVars and var not in otherVars:
						der[var][i1] = other.val[i1] * self.der[var][i1]
					else:
						der[var][i1] = self.val[i1] * other.der[var][i1]

		elif isinstance(other, (int, float)):
			# Multiplying the function by a constant, f(x) * c
			val = [v * other for v in self.val]

			# Multiply each partial by the constant
			for key in self.variables:
				der[key] = [d * other for d in self.der[key]]

		else:
			raise ValueError('Operand in multiplication is invalid. Operand must be an AD object or a number.')

		return AD(val, der, s, repr_s)

	def __rmul__(self, other):
		return self * other

	def __sub__(self, other):
		try:
			return self + -other
		except (TypeError, ValueError):
			raise ValueError('Operand in subtraction is invalid. Operand must be an AD object or a number.')

	def __rsub__(self, other):
		try:
			return -self + other
		except (TypeError, ValueError):
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
		val = [0] * self.num_vals
		der = {}
		s = '(' + str(self) + ') ** (' + str(other) + ')'
		repr_s = '(' + repr(self) + ') ** (' + repr(other) + ')'

		if isinstance(other, AD):
			myVars, otherVars, allVars = getVars(self.der, other.der)
			for v in allVars:
				der[v] = [0] * self.num_vals

			for i1 in range(self.num_vals):
				# Raising function to a function, f(x) ^ g(x)
				val[i1] = self.val[i1] ** other.val[i1]

				for var in allVars:
					if var in myVars and var in otherVars: # f(x, y) ^ g(x, y)
						der[var][i1] = (self.val[i1] ** (other.val[i1] - 1)) * (self.der[var][i1] * other.val[i1] + self.val[i1] * other.der[var][i1] * np.log(self.val[i1]))
					elif var in myVars and var not in otherVars: # f(x) ^ g(y)
						der[var][i1] = other.val[i1] * self.der[var][i1] * (self.val[i1] ** (other.val[i1] - 1))
					else: # f(y) ^ g(x)
						der[var][i1] = np.log(self.val[i1]) * other.der[var][i1] * (self.val[i1] ** other.val[i1])
		elif isinstance(other, (int, float)):
			# Raising the function to a constant, f(x) ^ c
			val = [v ** other for v in self.val]

			for v in self.variables:
				der[v] = [0] * self.num_vals

			for var in self.variables:
				for i1 in range(self.num_vals):
					der[var][i1] = other * self.der[var][i1] * (self.val[i1] ** (other - 1))

		else:
			raise ValueError('Operand in power is invalid. Operand must be an AD object or a number.')

		return AD(val, der, s, repr_s)

	def __rpow__(self, other):
		val = [0] * self.num_vals
		der = {}
		s = '(' + str(other) + ') ** (' + str(self) + ')'
		repr_s = '(' + repr(other) + ') ** (' + repr(self) + ')'

		if isinstance(other, (int, float)):
			for v in self.variables:
				der[v] = [0] * self.num_vals

			for i1 in range(self.num_vals):
				# Raising constant to the function, c ^ f(x)
				val[i1] = other ** self.val[i1]

				for var in self.variables:
					der[var][i1] = np.log(other) * self.der[var][i1] * (other ** self.val[i1])

		else:
			raise ValueError('Operand in power is invalid. Operand must be an AD object or a number.')

		return AD(val, der, s, repr_s)

	def __eq__(self, other):
		return type(self) == type(other) and self.val == other.val and self.der == other.der

	def __ne__(self, other):
		return not self == other

def getVars(der1, der2):
	# Get list of all variables of the two functions
	vars1 = set(der1.keys())
	vars2 = set(der2.keys())
	allVars = vars1.union(vars2)

	return vars1, vars2, allVars
