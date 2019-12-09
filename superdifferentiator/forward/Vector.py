import numpy as np

class Vector:
	def __init__(self, functions):
		self.functions = functions

	@property
	def val(self):
		# We reformat the output a little to make it easier for the user
		# Each element of the output list is the vector output of the function at that
		# index of given points

		vals = [v.val for v in self.functions]
		
		valsList = []
		for i1 in range(len(self.functions[0].val)):
			valsList.append(np.zeros((len(self.functions), 1)))

		for row in range(len(vals)):
			for col in range(len(vals[0])):
				valsList[col][row, 0] = vals[row][col]

		return valsList

	def jacobian(self):
		# Get the set of variables from the list of functions
		allVars = set()
		for f in self.functions:
			allVars = allVars.union(f.variables)

		# Make the list ordered
		allVars = list(allVars)

		# List for Jacobian matrix evaluated at multiple points
		jac = []
		for i1 in range(len(self.functions[0].val)):
			jac.append(np.zeros((len(self.functions), len(allVars))))

		for i1 in range(len(self.functions[0].val)):
			for row in range(len(self.functions)):
				f = self.functions[row]

				for col in range(len(allVars)):
					var = allVars[col]

					if var in f.variables:
						jac[i1][row, col] = f.der[var][i1]
					else:
						jac[i1][row, col] = 0

		return allVars, jac

	def euclidean_length(self):
		'''Calculates the euclidean, or L2 length of the function.
		NOTE: Only calculates for the first given values to evaluate the function at.
		'''
		l = 0
		for f in self.functions:
			l += f.val[0] ** 2
		l **= 0.5
		return l

	def __len__(self):
		return len(self.functions)

	def __eq__(self, other):
		if len(self) != len(other):
			return False

		for f, o in zip(self.functions, other.functions):
			if f != o:
				return False

		return True

	def __ne__(self, other):
		return not self == other

	def __lt__(self, other):
		try:
			return self.euclidean_length() < other.euclidean_length()
		except AttributeError:
			return self.euclidean_length() < other

	def __gt__(self, other):
		try:
			return self.euclidean_length() > other.euclidean_length()
		except AttributeError:
			return self.euclidean_length() > other

	def __le__(self, other):
		try:
			return self.euclidean_length() >= other.euclidean_length()
		except AttributeError:
			return self.euclidean_length() >= other

	def __ge__(self, other):
		try:
			return self.euclidean_length() >= other.euclidean_length()
		except AttributeError:
			return self.euclidean_length() >= other
