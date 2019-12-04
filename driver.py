from superdifferentiator.forward.functions import X

x = X(2, 'x')
y = X(2, 'y')
f = (5 * x) ** (2 * y)
print(f.val, f.der)
