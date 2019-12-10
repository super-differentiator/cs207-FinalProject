from superdifferentiator.forward.functions import X
import numpy as np

class GD:
    """
    The GD class allows the user to perform gradient descent on a function of their choice.
    User will pass in the function they defined and the initial value as an AD object to start
    gradient descent.
    """
    def __init__(self, max_iter=100, precision=0.001, step_size=0.01):
        self.max_iter = max_iter
        self.precision = precision
        self.step_size = step_size
        self.obj = []

    def cal_gradient_1d(self, f, x0):
        next_x = x0.val[0]
        for _ in range(self.max_iter):
            cur_x = next_x
            x = X(cur_x)
            val, der = f(x)
            self.obj.append(val[0])
            next_x = cur_x - self.step_size * der['x'][0]
            if abs(cur_x - next_x) <= self.precision:
                break

        return next_x

    def cal_gradient_2d(self, f, x0, y0):
        next_x = x0.val[0]
        next_y = y0.val[0]
        next_val = np.array([[next_x], [next_y]])

        for _ in range(self.max_iter):
            cur_val = next_val
            cur_x = float(cur_val[0][0])
            cur_y = float(cur_val[1][0])

            x = X(cur_x, 'x')
            y = X(cur_y, 'y')
            val, der = f(x, y)
            self.obj.append(val[0])
            der_vec = dic_to_vec(der)
            next_val = cur_val - self.step_size * der_vec
            if np.linalg.norm(cur_val - next_val) <= self.precision:
                break

        return next_val

    def cal_gradient_3d(self, f, x0, y0, z0):
        next_x = x0.val[0]
        next_y = y0.val[0]
        next_z = z0.val[0]
        next_val = np.array([[next_x], [next_y], [next_z]])

        for _ in range(self.max_iter):
            cur_val = next_val
            cur_x = float(cur_val[0][0])
            cur_y = float(cur_val[1][0])
            cur_z = float(cur_val[2][0])

            x = X(cur_x, 'x')
            y = X(cur_y, 'y')
            z = X(cur_z, 'z')
            val, der = f(x, y, z)
            self.obj.append(val[0])
            der_vec = dic_to_vec(der)
            next_val = cur_val - self.step_size * der_vec
            if np.linalg.norm(cur_val - next_val) <= self.precision:
                break

        return next_val

    def cal_gradient_nd(self, f, args):
        next_val = []
        for arg in args:
            next_val.append(arg.val[0])
        next_val = np.expand_dims(next_val, axis=0).T

        for _ in range(self.max_iter):
            cur_val = next_val
            var_list = [float(num[0]) for num in cur_val]
            input_list = [X(var_list[i], str(i)) for i in range(len(var_list))]
            val, der = f(input_list)
            self.obj.append(val[0])
            der_vec = dic_to_vec(der)
            next_val = cur_val - self.step_size * der_vec
            if np.linalg.norm(cur_val - next_val) <= self.precision:
                break

        return next_val


def dic_to_vec(dic):
    """
    convert dictionay of derivatives to a vector form
    """
    return np.array(list(dic.values()))