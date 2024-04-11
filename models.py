import numpy as np
import warnings


class Models:

    def __init__(self, X=None, y=None):
        """
        :param X (np.array): The attribute (feature) vector
        :param y (np.array): The label (response) vector
        """

        try:
            X = np.array(X)
            y = np.array(y)
        except Exception:
            raise TypeError("X or y is not or cannot be converted into a np.ndarray.")

        self.X = X
        self.y = y
        self._X_shape = X.shape
        self._y_shape = y.shape
        self._describe()

    def _describe(self):
        # TODO: describe X
        self._y_mean = np.mean(self.y)
        self._y_var = np.var(self.y, ddof=1)
        self._y_std = np.std(self.y, ddof=1)

    @property
    def X_shape_(self):
        return self._X_shape

    @property
    def y_shape_(self):
        return self._y_shape

    def fit(self):
        raise Exception("This is an empty model.")


class PredictionModels(Models):

    def __init__(self, X=None, y=None):
        super().__init__(X, y)
        self._SST = np.sum((y - self._y_mean) ** 2)
        self._SSR = None
        self._SSE = None
        self._X_test = None
        self._y_predict = None
        self._y_fit = None

    @property
    def SST_(self):
        return self._SST

    @property
    def SSE_(self):
        return self._SSE

    @property
    def SSR_(self):
        return self._SSR


class LinearRegression(PredictionModels):

    def __init__(self, X=None, y=None, fill_one=True, penalty='l2', C=0, max_iter=100, tol=1e-5):
        """
        :param X: The attribute (feature) vector
        :param y: The label (response) vector
        :param fill_one: Whether constants should be added to X
        :param penalty: The type of regularization
        :param C: Coefficient of regularization
        :param max_iter: Max iterating times for Lasso regression
        :param tol: Tolerance for convergence criteria
        """
        super().__init__(X, y)

        if fill_one:
            self._X = np.hstack((np.ones((self._X_shape[0], 1)), self.X))
            self._X_shape = self._X.shape
        else:
            self._X = X
        self._X_T = X.T
        self._X_T_X = self._X_T @ self._X
        self._X_T_y = self._X_T @ self.y

        if penalty == "l1":
            self.penalty = 1
        elif penalty == "l2":
            self.penalty = 2

        self.C = C
        self._beta = None
        self.max_iter = max_iter
        self.tol = tol

    def fit(self):

        if self.penalty == 2:
            self._beta = np.linalg.inv(self._X_T_X + self.C * np.eye(self._X_shape[1])) @ self._X_T_y
        else:
            is_converge = False
            self._beta = np.zeros(self._X_shape[1])
            for _ in range(self.max_iter):
                start_beta = self._beta.copy()
                for k in range(self._X_shape[1]):
                    best_beta_k = self._best_beta_k(k)
                    self._beta[k] = best_beta_k
                if np.max(self._beta - start_beta) < self.tol:
                    is_converge = True
                    break
            if not is_converge:
                warnings.warn("The model doesn't converge")

        self._y_fit = self._X @ self._beta
        self._SSR = np.sum((self._y_fit - self.y) ** 2)
        self._SSE = self._SST - self._SSR

    def predict(self, X_test, fill_one=True):
        if not self._beta:
            raise Exception("This model has not fitted yet.")

        if fill_one:
            self._X_test = np.hstack(np.ones(X_test.shape[0], X_test))
        else:
            self._X_test = X_test
        self._y_predict = X_test @ self._beta
        return self._y_predict

    def _best_beta_k(self, k):
        return self._sub_gradient(self._X_T_y[k][0] - self._X_T_X[k] @ self._beta + self._X_T_X[k][k] * self._beta[k], k)

    def _sub_gradient(self, K, k):
        if K > 0.5 * self.C:
            return (K - 0.5 * self.C) / self._X_T_X[k][k]
        if K < - 0.5 * self.C:
            return (K + 0.5 * self.C) / self._X_T_X[k][k]
        return 0

    @property
    def coef_(self, index=None):
        if index is None:
            return self._beta
        else:
            return self._beta[index]


class ClassificationModels(Models):

    def __init__(self, X=None, y=None):
        super().__init__(X, y)


class LogisticRegression(ClassificationModels):

    def __init__(self, X=None, y=None):
        super().__init__(X, y)

    def fit(self):
        # TODO
        pass

    def _gradient_descent(self):
        # TODO
        pass
