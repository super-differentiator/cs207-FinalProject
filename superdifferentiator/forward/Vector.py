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
