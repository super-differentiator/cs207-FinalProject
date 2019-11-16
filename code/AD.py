import numpy as np


class AD:
    """

    """

    def __init__(self):
        self.val = 0
        self.der = 0

    def __add__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        newAutoDiff = AD()

        if isinstance(other, AD):
            newAutoDiff.val = self.val + other.val
            newAutoDiff.der = self.der + other.der

        elif isinstance(other, (int, float)):
            newAutoDiff.val = self.val + other
            newAutoDiff.der = self.der

        else:
            raise ValueError('operand in addition is invalid')
        return newAutoDiff

    def __radd__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        return self.__add__(other)

    def __sub__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        newAutoDiff = AD()
        if isinstance(other, AD):
            newAutoDiff.val = self.val - other.val
            newAutoDiff.der = self.der - other.der

        elif isinstance(other, (int, float)):
            newAutoDiff.val = self.val - other
            newAutoDiff.der = self.der
        else:
            raise ValueError('operand in subtraction is invalid')
        return newAutoDiff

    def __rsub__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        newAutoDiff = AD()
        if isinstance(other, (int, float)):
            newAutoDiff.val = other - self.val
            newAutoDiff.der = self.der * -1

        else:
            raise ValueError('operand in subtraction is invalid')
        return newAutoDiff

    def __mul__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        newAutoDiff = AD()

        if isinstance(other, AD):
            newAutoDiff.val = self.val * other.val
            newAutoDiff.der = self.val * other.der + self.der * other.val
        elif isinstance(other, (int, float)):
            newAutoDiff.val = self.val * other
            newAutoDiff.der = self.der * other
        else:
            raise ValueError('operand in multiplication is invalid')
        return newAutoDiff

    def __rmul__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        return self.__mul__(other)

    def __div__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        newAutoDiff = AD()
        if isinstance(other, AD):
            newAutoDiff.val = self.val / other.val
            newAutoDiff.der = (other.val * (self.der) + self.val * (other.der)) / other.val ** 2

        elif isinstance(other, (int, float)):
            newAutoDiff.val = self.val / other
            newAutoDiff.der = self.der / other

        else:
            raise ValueError('operand in division is invalid')
        return newAutoDiff

    def __rdiv__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        if isinstance(other, (int, float)):
            newAutoDiff = other * self ** -1
        else:
            raise ValueError('operand in division is invalid')
        return newAutoDiff

    def __pow__(self, num):
        """

        Parameters
        ----------
        num

        Returns
        -------

        """
        if isinstance(num, int):
            newAutoDiff = AD()
            newAutoDiff.val = self.val ** num
            newAutoDiff.der = num * self.val ** (num - 1)
        else:
            raise AttributeError('exponential has to be an integer')
        return newAutoDiff

    def __rpow__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        newAutoDiff = AD()

        newAutoDiff.val = other ** self.val
        newAutoDiff.der = other ** self.val * np.log(other)
        return newAutoDiff

    def __neg__(self):
        newAutoDiff = AD()
        newAutoDiff.val = -self.val
        newAutoDiff.der = -self.der
        return newAutoDiff

    def __str__(self):
        """

        Returns
        -------

        """
        return 'val = {}, der={}'.format(self.val, self.der)


class X(AD):
    """

    """

    def __init__(self, alpha):
        # Initialize as f(x) = x
        self.val = alpha
        self.der = 1.0


class Sin(AD):
    """

    """

    def __init__(self, fun):
        self.val = np.sin(fun.val)
        self.der = fun.der * np.cos(fun.val)


class Cos(AD):
    """

    """

    def __init__(self, fun):
        self.val = np.cos(fun.val)
        self.der = fun.der * -np.sin(fun.val)


class Tan(AD):
    """

    """

    def __init__(self, fun):
        self.val = np.tan(fun.val)
        self.der = fun.der / (np.cos(fun.val) ** 2)


class Log(AD):
    """

    """

    def __init__(self, fun):
        self.val = np.log(fun.val)
        self.der = fun.der / fun.val


class Exp(AD):
    """

    """

    def __init__(self, fun):
        self.val = np.exp(fun.val)
        self.der = fun.der * np.exp(fun.val)
