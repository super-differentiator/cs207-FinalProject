from superdifferentiator.forward.functions import X
from superdifferentiator.additional_features.gradient_descent import GD
import numpy as np


class LinearRegression:
    """
    This class is a demo of how our gradient descent method can be used in training machine learning model.
    Although linear regression is a well understood algorithm and there is analytical solution to linear regression,
    what we are trying to show here is that as long as we define an objective/loss function that we want to minimize for,
    we can just replace this new objective function with _linear_obj() in this class and it will become a new machine
    learning algorithm

    """

    def __init__(self, max_iter=10000):
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept = None
        self.x = None
        self.y = None

    def _linear_obj(self, ws):
        """
        objective function for linear regression
        """
        fx = 0
        for i in range(self.x.shape[0]):
            cur = 0
            for j in range(len(ws)):
                cur = cur + self.x[i][j] * ws[j]
            fx = fx + (self.y[i] - cur) * (self.y[i] - cur)

        fx = fx / self.x.shape[0]

        # idx = np.random.randint(low=0, high=self.x.shape[0] ,size=1)[0]
        # xc = self.x[idx]
        # yc = self.y[idx]
        # print(xc)
        # print(yc)
        #
        # fx = 0
        # cur = 0
        # for i in range(len(xc)):
        #     cur += xc[i] * ws[i]
        #     print(cur)
        # fx = (yc - cur)
        # fx = fx*fx

        return fx.val, fx.der

    def fit(self, x, y):
        # add a column of 1s for intercept
        x1s = np.ones((x.shape[0], 1))
        x = np.hstack((x, x1s))
        self.x = x
        self.y = y

        num_feature = x.shape[1]

        # list of coefficients
        wad = [X(100, str(i)) for i in range(num_feature)]

        grad = GD(max_iter=self.max_iter, step_size=0.001, precision=0.00001)
        res = grad.cal_gradient_nd(self._linear_obj, wad)
        # print(grad.obj)
        self.coef_ = res
        self.intercept = self.coef_[-1][0]

    def get_params(self):
        return self.coef_.T[0]

    def predict(self, x):
        # add a column of 1s for intercept
        x1s = np.ones((x.shape[0], 1))
        x = np.hstack((x, x1s))
        return x @ self.coef_

    def score(self, x, y):
        """
        r^2 score
        """
        yhat = self.predict(x)
        yhat = np.array(yhat).T[0]
        # print('hello')
        # print(yhat)
        # print('hello')
        ybar = np.mean(y)
        ssreg = np.sum((y - yhat) ** 2)
        sstot = np.sum((y - ybar) ** 2)
        return 1 - ssreg / sstot
        # return np.mean((y - y_pred) ** 2)