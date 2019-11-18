# CS207 Final Project

import numpy as np

class AD:
	'''The AD class stores the function value and the derivative for a function f(x),
	as well as the derivative rules for addition, subtraction, multiplication, division,
	and raising to a power, by overloading the +, -, *, /, and ** operators.
	'''

	def __init__(self, val, der):
		self._val = val
		self._der = der

	@property
	def val(self):
		return self._val

	@property
	def der(self):
		return self._der

	def __add__(self, other):
		v = 0
		d = 0

		if isinstance(other, AD):
			# Adding two functions, f(x) + g(x)
			v = self.val + other.val
			d = self.der + other.der
		elif isinstance(other, (int, float)):
			# Adding a function and scalar, f(x) + c
			v = self.val + other
			d = self.der
		else:
			raise ValueError('Operand in addition is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __radd__(self, other):
		return self + other

	def __mul__(self, other):
		v = 0
		d = 0

		if isinstance(other, AD):
			# Multiplying two functions, f(x) * g(x)
			v = self.val * other.val
			d = self.val * other.der + self.der * other.val
		elif isinstance(other, (int, float)):
			# Multiplying the function by a constant, f(x) * c
			v = self.val * other
			d = self.der * other
		else:
			raise ValueError('Operand in multiplication is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __rmul__(self, other):
		return self * other

	def __sub__(self, other):
		v = 0
		d = 0

		if isinstance(other, AD):
			# Subtracting two functions, f(x) - g(x)
			v = self.val - other.val
			d = self.der - other.der
		elif isinstance(other, (int, float)):
			# Subtracting a constant from the function, f(x) - c
			v = self.val - other
			d = self.der
		else:
			raise ValueError('Operand in subtraction is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __rsub__(self, other):
		v = 0
		d = 0

		# Subtracting the function from a constant, c - f(x)
		if isinstance(other, (int, float)):
			v = other - self.val
			d = -self.der
		else:
			raise ValueError('Operand in subtraction is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __neg__(self):
		return (-1) * self

	def __truediv__(self, other):
		v = 0
		d = 0

		if isinstance(other, AD):
			# Dividing two functions, f(x) / g(x)
			v = self.val / other.val
			d = (other.val * self.der - self.val * other.der) / (other.val ** 2)
		elif isinstance(other, (int, float)):
			# Dividing the function by a constant, f(x) / c
			v = self.val / other
			d = self.der / other
		else:
			raise ValueError('Operand in division is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __rtruediv__(self, other):
		v = 0
		d = 0

		# Dividing a constant by the function, c / f(x)
		if isinstance(other, (int, float)):
			v = other / self.val
			d = -other * self.der / (self.val ** 2)
		else:
			raise ValueError('Operand in division is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __pow__(self, other):
		v = 0
		d = 0

		if isinstance(other, AD):
			# Raising function to a function, f(x) ^ g(x)
			v = self.val ** other.val
			d = self.val ** (other.val - 1) * (other.val * self.der + self.val * np.log(self.val) * other.der)
		elif isinstance(other, (int, float)):
			# Raising the function to a constant, f(x) ^ c
			v = self.val ** other
			d = other * self.val ** (other - 1) * self.der
		else:
			raise ValueError('Operand in power is invalid. Operand must be an AD object or a number.')

		return AD(v, d)

	def __rpow__(self, other):
		v = 0
		d = 0

		if isinstance(other, (int, float)):
			# Raising constant to the function, c ^ f(x)
			v = other ** self.val
			d = np.log(other) * other ** self.val * self.der
		else:
			raise ValueError('Operand in power is invalid. Operand must be an AD object or a number.')

		return AD(v, d)
