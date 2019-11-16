from AutoDifferentiation.forward.AD import X, Sin, Cos, Ln
x = X(7.2)
fx = Ln(Cos(Sin((x ** 2) + (3 * x) - 4)))
minus_x = -x

print(fx.val, fx.der)
print(minus_x.val, minus_x.der)
