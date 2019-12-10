from superdifferentiator.forward.functions import X
from superdifferentiator.additional_features.gradient_descent import GD
import numpy as np


x = X(4)

fx = (3 - x)*(3-x)
print(fx)