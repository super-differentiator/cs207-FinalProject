import numpy as np

class AD:
	def __init__(self):
		self.val = 0
		self.der = 0

	def __add__(self, other):
		newAutoDiff = AD()

		try:
			newAutoDiff.val = self.val + other.val
			newAutoDiff.der = self.der + other.der
		except AttributeError:
			newAutoDiff.val = self.val + other
			newAutoDiff.der = self.der

		return newAutoDiff

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		newAutoDiff = AD()

		try:
			newAutoDiff.val = self.val * other.val
			newAutoDiff.der = self.val * other.der + self.der * other.val
		except AttributeError:
			newAutoDiff.val = self.val * other
			newAutoDiff.der = self.der * other

		return newAutoDiff

	def __rmul__(self, other):
		return self.__mul__(other)

class X(AD):
	def __init__(self, alpha):
		# Initialize as f(x) = x
		self.val = alpha
		self.der = 1.0

class Sin(AD):
	def __init__(self, fun):
		self.val = np.sin(fun.val)
		self.der = fun.der * np.cos(fun.val)

class Cos(AD):
	def __init__(self, fun):
		self.val = np.cos(fun.val)
		self.der = fun.der * -np.sin(fun.val)

class Tan(AD):
	def __init__(self, fun):
		self.val = np.tan(fun.val)
		self.der = fun.der / (np.cos(fun.val) ** 2)

class Ln(AD):
	def __init__(self, fun):
		self.val = np.log(fun.val)
		self.der = fun.der / fun.val

class Exp(AD):
	def __init__(self, fun):
		self.val = np.exp(fun.val)
		self.der = fun.der * np.exp(fun.val)

def main():
	x = X(3)
	fx = 3 * x * x + 2 * x + 5
	print('f(x) = 3x^2 + 2x + 5:', fx.val, fx.der)

	sinx = Sin(fx)
	print('sin[f(x)]:', sinx.val, sinx.der)

	cosx = Cos(fx)
	print('cos[f(x)]:', cosx.val, cosx.der)

	tanx = Tan(fx)
	print('tan[f(x)]:', tanx.val, tanx.der)

	lnx = Ln(fx)
	print('ln[f(x)]:', lnx.val, lnx.der)

	expx = Exp(fx)
	print('Exp[f(x)]:', expx.val, expx.der)

if __name__ == '__main__':
	main()
