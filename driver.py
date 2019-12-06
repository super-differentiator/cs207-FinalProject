from superdifferentiator.forward.functions import X, Log
from superdifferentiator.forward.Vector import Vector

x = X(3, 'x')
y = X(4, 'y')
fx = x * x * y * y
allVars, jacs = fx.jacobian()
print('f(x, y) = x^2 * y^2 for f(3, 4)')
print(allVars)
print(jacs[0])
print()

xV = [3, 5]
x = X(xV)
f1 = (x ** 2) + (2 * x) + 3
f2 = (2 ** x)
v = Vector([f1, f2])
vals = v.val
allVars, jacs = v.jacobian()

print('g(x) = x^2 + 2x + 3 for x = 3, 5 =', f1.val)
print('h(x) = 2^x for x = 3, 5 =', f2.val)
print()

print('f(x) = [x^2 + 2x + 3, 2^x]\'')

for i1 in range(len(xV)):
	print('f(', xV[i1], ') = ', sep = '')
	print(vals[i1])
	print()
	print('Jacobian of f for x =', xV[i1], end = ':\n')
	print(jacs[i1])
	print()
