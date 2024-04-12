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
            if y.ndim == 1:
                y = y[:, np.newaxis]
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
        self.R_squared = None
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

    def __init__(self, X, y, fill_one=True, penalty='l2', C=0, max_iter=100, tol=1e-5):
        """
        Initialize the linear regression model with given parameters.

        The optimization objective for lasso is::

            ||y - Xw||^2_2 + C * ||w||_1

        C corresponds to `alpha = C / (2 * sample_size)` in other lasso models like
        sklearn.linear_model.Lasso.

        The optimization for lasso is done by subgradient coordinate descent.

        The optimization objective for ridge is::

            ||y - Xw||^2_2 + C * ||w||_2

        C is the same as alpha in other ridge models like sklearn.linear_model.Ridge.

        To fit a multiple linear regression model without regularization, please use 'l2'
        penalty and set `C = 0`.

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
            self._X = self.X
        self._X_T = self._X.T
        self._X_T_X = self._X_T @ self._X
        self._X_T_y = self._X_T @ self.y

        if penalty == "l1":
            self.penalty = 1
        elif penalty == "l2":
            self.penalty = 2
        else:
            raise NotImplementedError(f'{penalty} penalty has not been implemented')

        self.C = C
        self._coef = None
        self.max_iter = max_iter
        self.tol = tol

    def fit(self):

        if self.penalty == 2:
            self._coef = np.linalg.inv(self._X_T_X + self.C * np.eye(self._X_shape[1])) @ self._X_T_y
        else:
            is_converge = False
            self._coef = np.zeros(self._X_shape[1])
            for _ in range(self.max_iter):
                start_coef = self._coef.copy()
                for k in range(self._X_shape[1]):
                    best_coef_k = self._best_coef_k(k)
                    self._coef[k] = best_coef_k
                if np.max(self._coef - start_coef) < self.tol:
                    is_converge = True
                    break
            if not is_converge:
                warnings.warn("The model doesn't converge")

        self._y_fit = self._X @ self._coef
        self._SSR = np.sum((self._y_fit - self.y) ** 2)
        self._SSE = self._SST - self._SSR
        self.R_squared = self._SSR / self._SST

    def predict(self, X_test, fill_one=True):
        if not self._coef:
            raise Exception("This model has not fitted yet.")

        if fill_one:
            self._X_test = np.hstack(np.ones(X_test.shape[0], X_test))
        else:
            self._X_test = X_test
        self._y_predict = X_test @ self._coef
        return self._y_predict

    def _best_coef_k(self, k):
        return self._sub_gradient(self._X_T_y[k][0] - self._X_T_X[k] @ self._coef +
                                  self._X_T_X[k][k] * self._coef[k], k)

    def _sub_gradient(self, K, k):
        if K > 0.5 * self.C:
            return (K - 0.5 * self.C) / self._X_T_X[k][k]
        if K < - 0.5 * self.C:
            return (K + 0.5 * self.C) / self._X_T_X[k][k]
        return 0

    @property
    def coef_(self, index=None):
        """
        :param index: A specific index in the coefficient vector
        :return: If index is specified, returns the coefficient value at that index.
                             Otherwise, returns the entire coefficient vector.
        """
        if index is None:
            return self._coef
        else:
            return self._coef[index]


class ClassificationModels(Models):

    def __init__(self, X=None, y=None, pre_process=False):
        super().__init__(X, y)
        self._y = None

        if pre_process:
            self._preprocess_y()
        else:
            self._y = self.y

        self._class_num = len(np.unique(self._y))

    def _preprocess_y(self):
        y_dict = dict()
        for i, y in enumerate(self.y):
            if y[0] not in y_dict:
                self._y[i][0] = len(y_dict)
                y_dict[y[0]] = len(y_dict)
            else:
                self._y[i][0] = y_dict[y[0]]
    
    @property
    def class_num_(self):
        return self._class_num

    
class LogisticRegression(ClassificationModels):

    def __init__(self, X, y, fill_one=True, pre_process=False, lr=0.04, penalty='l2', C=0, max_iter=10000, tol=1e-5,
                 batch='all'):
        """
        Initialize the linear regression model with given parameters.

        The objective function for l2 regulariztion is::

            l(w) + C / 2 * ||w||_2^2

        The optimization is done by gradient descent.

        :param X: The attribute (feature) vector
        :param y: The label (response) vector
        :param fill_one: Whether constants should be added to X
        :param pre_process: Whether to preprocess vector y
        :param lr: The learning rate for gradient descent
        :param penalty: The type of regularization
        :param C: Coefficient of regularization
        :param max_iter: Max iterating times for Lasso regression
        :param tol: Tolerance for convergence criteria
        :param batch: The batch size for gradient descent
        """
        super().__init__(X, y, pre_process=pre_process)

        if fill_one:
            self._X = np.hstack((np.ones((self._X_shape[0], 1)), self.X))
            self._X_shape = self._X.shape
        else:
            self._X = self.X

        if penalty == 'l2':
            self.penalty = 2
        elif penalty is None:
            self.penalty = 2
            self.C = 0
        else:
            raise NotImplementedError(f'{penalty} penalty has not been implemented.')

        self.C = C
        self.max_iter = max_iter
        self.lr = lr

        if self._class_num == 1:
            raise Exception('There is only one group of y')
        elif self._class_num == 2:
            self._coef = np.zeros(self._X_shape[1])
            self._type = 'Binary model'
        else:
            self._coef = np.zeros((self._class_num - 1, self._X_shape[1]))
            self._type = 'Multiclass model'

        if batch == 'all' or batch >= self._X_shape[0]:
            self.batch = self._X_shape[0]
        elif batch < 1:
            self.batch = 1
            warnings.warn('Batch must be a positive number and is automatically set to 1')
        else:
            self.batch = int(batch)

        self.tol = tol

    def fit(self):
        if self._class_num == 2:
            self._binary_solver()
        else:
            self._multiclass_solver()

    def _binary_solver(self):
        for _ in range(self.max_iter):
            curr_coef = self._coef.copy()
            self._binary_gradient_descent()
            if np.max(abs(curr_coef - self._coef)) < self.tol:
                break

    def _binary_gradient_descent(self):
        index_list = np.random.choice(np.arange(self._X_shape[0]), self.batch, replace=False)
        self._coef += self.lr * (sum(
            self._X[i] * (self._y[i][0] - self._sigmoid(np.dot(self._X[i], self._coef)))
            for i in index_list
        ) - self.C * self._coef)

    def _multiclass_solver(self):
        # TODO
        pass

    def predict(self, X, threshold=0.5, fill_one=True):
        if self._class_num == 2:
            res = self.predict_prob(X, fill_one=fill_one)
            for i in range(len(res)):
                res[i] = 0 if res[i] < threshold else 1
            return res
        else:
            # TODO
            pass

    def predict_prob(self, X, fill_one=True):
        if self._class_num == 2:
            res = [0 for _ in range(X.shape[0])]

            if fill_one:
                X = np.hstack((np.ones((X.shape[0], 1)), X))

            for i, x in enumerate(X):
                res[i] = self._sigmoid(np.dot(x, self._coef))
            return res
        else:
            # TODO
            pass

    @property
    def coef_(self):
        return self._coef

    @property
    def type_(self):
        """
        :return: The type of the logistic regression model, i.e., binary model or multiclass model
        """
        return self._type

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.e ** -x)
