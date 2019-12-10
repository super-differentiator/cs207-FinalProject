from superdifferentiator.forward.functions import Sin,X
from superdifferentiator.forward.Vector import Vector
from superdifferentiator.additional_features.newton_method import Newton


def test_newton1():
    
    f1 = lambda x: Sin(x[0])

    f = [f1]
    init_x = [1.0]
    n = Newton(f, init_x)
    
    
    x = X(n[0])
    f11= Sin(x)

    
    assert f11.val[0] < 1e-8
