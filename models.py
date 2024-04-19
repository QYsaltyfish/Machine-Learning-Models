import numpy as np
import warnings


class SupervisedModels:

    def __init__(self):

        self.X = None
        self.y = None
        self._X_shape = None
        self._y_shape = None

    def _describe(self):
        self._X_mean = np.mean(self.X, axis=0)
        self._X_var = np.var(self.X, axis=0, ddof=1)
        self._X_std = np.std(self.X, axis=0, ddof=1)
        self._X_shape = self.X.shape
        self._y_shape = self.y.shape
        self._y_mean = np.mean(self.y)
        self._y_var = np.var(self.y, ddof=1)
        self._y_std = np.std(self.y, ddof=1)

    def _preprocess(self, X, y):
        try:
            X = np.array(X)
            y = np.array(y)
            if y.ndim == 1:
                y = y[:, np.newaxis]
        except Exception:
            raise TypeError("X or y is not or cannot be converted into a np.ndarray.")
        self.X = X
        self.y = y
        self._describe()

    @property
    def X_shape_(self):
        return self._X_shape

    @property
    def y_shape_(self):
        return self._y_shape

    def fit(self, X, y):
        raise Exception("This is an empty model.")


class PredictionModels(SupervisedModels):

    def __init__(self, fill_one=False):
        super().__init__()
        self.fill_one = fill_one
        self._SST = None
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

    def _preprocess(self, X, y):
        super()._preprocess(X, y)
        if self.fill_one:
            self._X = np.hstack((np.ones((self._X_shape[0], 1)), self.X))
            self._X_shape = self._X.shape
        else:
            self._X = self.X
        self._SST = np.sum((y - self._y_mean) ** 2)


class LinearRegression(PredictionModels):

    def __init__(self, fill_one=True, penalty='l2', C=0, max_iter=100, tol=1e-5):
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

        :param fill_one: Whether constants should be added to X
        :param penalty: The type of regularization
        :param C: Coefficient of regularization
        :param max_iter: Max iterating times for Lasso regression
        :param tol: Tolerance for convergence criteria
        """

        super().__init__(fill_one=fill_one)

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

    def fit(self, X, y):
        self._preprocess(X, y)

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

    def _preprocess(self, X, y):
        super()._preprocess(X, y)
        self._X_T = self._X.T
        self._X_T_X = self._X_T @ self._X
        self._X_T_y = self._X_T @ self.y

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


class ClassificationModels(SupervisedModels):

    def __init__(self, pre_process=False, fill_one=False):
        super().__init__()
        self._y = None
        self.pre_process = pre_process
        self.fill_one = fill_one
        self._class_num = None

    def _preprocess_y(self):
        y_dict = dict()
        for i, y in enumerate(self.y):
            if y[0] not in y_dict:
                self._y[i][0] = len(y_dict)
                y_dict[y[0]] = len(y_dict)
            else:
                self._y[i][0] = y_dict[y[0]]
        return len(y_dict)
    
    @property
    def class_num_(self):
        return self._class_num

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.fill_one:
            self._X = np.hstack((np.ones((self._X_shape[0], 1)), self.X))
            self._X_shape = self._X.shape
        else:
            self._X = self.X

    
class LogisticRegression(ClassificationModels):

    def __init__(self, fill_one=True, pre_process=False, lr=0.04, penalty='l2', C=0, max_iter=10000, tol=1e-5,
                 batch='all'):
        """
        Initialize the linear regression model with given parameters.

        The objective function for l2 regulariztion is::

            l(w) + C / 2 * ||w||_2^2

        The optimization is done by gradient descent.

        :param fill_one: Whether constants should be added to X
        :param pre_process: Whether to preprocess vector y
        :param lr: The learning rate for gradient descent
        :param penalty: The type of regularization
        :param C: Coefficient of regularization
        :param max_iter: Max iterating times for Lasso regression
        :param tol: Tolerance for convergence criteria
        :param batch: The batch size for gradient descent
        """
        super().__init__(pre_process=pre_process, fill_one=fill_one)

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
        self.batch = batch
        self.tol = tol

    def fit(self, X, y):
        self._preprocess(X, y)
        if self._class_num == 2:
            self._binary_solver()
        else:
            self._multiclass_solver()

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.pre_process:
            self._y = np.zeros_like(self.y)
            self._class_num = self._preprocess_y()
        else:
            self._y = self.y
            self._class_num = len(np.unique(self._y))

        if self._class_num == 1:
            raise Exception('There is only one group of y')
        elif self._class_num == 2:
            self._coef = np.zeros(self._X_shape[1])
        else:
            self._coef = np.zeros((self._class_num - 1, self._X_shape[1]))

        if self.batch == 'all' or self.batch >= self._X_shape[0]:
            self._batch = self._X_shape[0]
        elif self.batch < 1:
            self._batch = 1
            warnings.warn('Batch must be a positive number and is automatically set to 1')
        else:
            self._batch = int(self.batch)

    def _binary_solver(self):
        for _ in range(self.max_iter):
            curr_coef = self._coef.copy()
            self._binary_gradient_descent()
            if np.max(abs(curr_coef - self._coef)) < self.tol:
                break

    def _binary_gradient_descent(self):
        index_list = np.random.choice(np.arange(self._X_shape[0]), self._batch, replace=False)
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

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.e ** -x)


class SVM(ClassificationModels):

    def __init__(self, pre_process=True, margin='soft', C=1, eps=1e-3):
        """
        Initialize the SVM model with given parameters.

        The SVM is solved by Sequential Minimal Optimization(SMO) algorithm.

        Note1: Hard-margin SVM hasn't been done yet and please use margin='soft', C=np.inf.
        If you intend to use hard-margin, please ensure that the data is perfectly linearly separable;
        otherwise, the program may enter an infinite loop.

        Note2: The heuristic function has not been actually utilized and may be employed in future version
        to enhance efficiency.

        :param pre_process: Whether to preprocess vector y
        :param margin: The type of margin of SVM model (hard or soft)
        :param C: Coefficient for soft-margin
        :param eps: Tolerance for convergence criteria
        """
        super().__init__()
        self.pre_process = pre_process

        if margin == 'hard':
            self._margin = 0
            warnings.warn("Hard-margin SVM hasn't been done yet")
        elif margin == 'soft':
            self._margin = 1
        else:
            raise Exception(f"Unexpected margin type: {margin}")

        self.C = C
        self.eps = eps
        self._b = 0

    def _preprocess_y(self):
        y_dict = dict()

        if any(self._y == [0]):
            for y in self._y:
                if y[0] == 0:
                    y[0] = -1
            return

        if any(self._y == [-1]):
            for y in self._y:
                if y[0] != -1:
                    y[0] = 1
            return

        for i, y in enumerate(self._y):
            if y[0] not in y_dict:
                if len(y_dict) == 0:
                    y_dict[y[0]] = -1
                else:
                    y_dict[y[0]] = 1
            self._y[i][0] = y_dict[y[0]]

    def fit(self, X, y):
        self._preprocess(X, y)

        if self._margin == 0:
            self._hard_smo_solver()
        elif self._margin == 1:
            self._soft_smo_solver()

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.pre_process:
            self._y = np.zeros_like(self.y)
            self._preprocess_y()
            self._class_num = 2
        else:
            self._y = self.y
            self._class_num = len(np.unique(self._y))

        if self._class_num == 1:
            raise Exception('There is only one group of y')
        elif self._class_num == 2:
            self._coef = np.zeros(self._X_shape[1])
            self._intercept = 0
        else:
            raise NotImplementedError("SVM doesn't support multiple classes of y.")

        self._X_dot = np.zeros((self._X_shape[0], self._X_shape[0]))
        self._d = np.zeros(self._X_shape[0])
        self._E = np.zeros(self._X_shape[0])
        self._alpha = np.zeros(self._X_shape[0])

        for i in range(self._X_shape[0]):
            for j in range(self._X_shape[0]):
                self._X_dot[i][j] = np.dot(self.X[i], self.X[j])

        for i in range(self._X_shape[0]):
            self._E[i] = -self._y[i][0]

    def _hard_smo_solver(self):
        # TODO
        pass

    def _soft_smo_solver(self):
        # TODO: Apply the heuristic function
        # Repeat the traversal pattern until no sample violates KKT conditions
        while True:
            # First traversal, traverse the whole training set and find the first sample violating KKT conditions.
            find_violator = False
            for i in range(self._X_shape[0]):
                if self._check_epsilon_kkt_violation(i):
                    find_violator = True
                    l = list(range(self._X_shape[0]))
                    j = np.random.choice(l[:i] + l[i + 1:])
                    # j = self._find_second_alpha(i)
                else:
                    continue
                new_alpha2, L, H = self._clip_alpha2(i, j, self._get_unclipped_alpha2(i, j))
                new_alpha1 = self._alpha[i] + self._y[i][0] * self._y[j][0] * (self._alpha[j] - new_alpha2)
                new_b = self._get_new_b(i, j, new_alpha1, new_alpha2, L, H)
                self.update_E(i, j, new_alpha1, new_alpha2, new_b)
                self._b = new_b
                self._alpha[i] = new_alpha1
                self._alpha[j] = new_alpha2

            # No violators and we break
            if not find_violator:
                break
            find_violator = False

            # Find all the non-bound samples
            non_bound_indexs = []
            for i in range(self._X_shape[0]):
                if 0 < self._alpha[i] < self.C: non_bound_indexs.append(i)

            # Second traversal, traverse all the non-bound samples and optimize, until all non-bound samples
            # satisfy KKT conditions.
            while True:
                find_violator_this_loop = False
                for i in non_bound_indexs:
                    if self._check_epsilon_kkt_violation(i):
                        find_violator_this_loop = True
                        find_violator = True
                        l = list(range(self._X_shape[0]))
                        j = np.random.choice(l[:i] + l[i + 1:])
                        # j = self._find_second_alpha(i)
                    else:
                        continue
                    new_alpha2, L, H = self._clip_alpha2(i, j, self._get_unclipped_alpha2(i, j))
                    new_alpha1 = self._alpha[i] + self._y[i][0] * self._y[j][0] * (self._alpha[j] - new_alpha2)
                    new_b = self._get_new_b(i, j, new_alpha1, new_alpha2, L, H)
                    self.update_E(i, j, new_alpha1, new_alpha2, new_b)
                    self._b = new_b
                    self._alpha[i] = new_alpha1
                    self._alpha[j] = new_alpha2
                if not find_violator_this_loop:
                    break

            # No violators and we break
            if not find_violator:
                break

        self._coef = sum((self._alpha[i] * self._y[i][0] * self.X[i] for i in range(self._X_shape[0])))
        b_star = [self._y[j][0] -
                  sum(self._alpha[i] * self._y[i] * self._X_dot[i][j]
                      for i in range(self._X_shape[0]) if self._alpha[i] != 0)
                  for j in range(self._X_shape[0]) if 0 < self._alpha[j] < self.C]
        self._intercept = np.mean(b_star)

    def _check_epsilon_kkt_violation(self, i):
        """
        Check the KKT conditions within precision epsilon.

        If     alpha_i = 0, then 1 - eps <= y_i * d_i
        If 0 < alpha_i < C, then 1 - eps <= y_i * d_i <= 1 + eps
        If     alpha_i = C, then            y_i * d_i <= 1 + eps

        :param i: The index of the coefficient to check KKT condition.
        :return: Whether the coefficient **violates** epsiolon-KKT condition.
        """
        return (self._alpha[i] < self.C and self._y[i][0] * self._E[i] < -self.eps
                or
                self._alpha[i] > 0 and self._y[i][0] * self._E[i] > self.eps)

    def _find_second_alpha(self, i):
        """
        After selecting the first variable alpha1 to optimize, choose the second variable alpha2 using heuristic
        function.

        The heuristic function is::

            h(i, j) = |E_i - E_j|

        :param i: The index of the first alpha
        :return: The index of the second alpha
        """
        return max((j for j in range(self._X_shape[0]) if j != i), key=lambda j: abs(self._E[i] - self._E[j]))

    def _get_unclipped_alpha2(self, i, j):
        """
        Get the unclipped value of alpha_j by optimizting both alpha_i and alpha_j in the dual problem.

        The update rule is::

            alpha_j (unclipped) = alpha_j (old) + y_2 * (E_i - E_j) / (k_ii + k_jj - 2k_ij)

        :param i: The index of the first alpha
        :param j: The index of the second alpha
        :return: Unclipped alpha_j
        """
        eta = self._X_dot[i][i] + self._X_dot[j][j] - 2 * self._X_dot[i][j]
        if eta > 0:
            return self._alpha[j] + self._y[j][0] * (self._E[i] - self._E[j]) / eta
        else:
            return None

    def _clip_alpha2(self, i, j, unclipped_alpha2):
        """
        Get the clipped value of alpha_j.

        The upper bound of alpha_j, H, is::

            H = min(C, alpha_i + alpha_j),     if y_i == y_j
                min(C, alpha_j - alpha_i + C), if y_i != y_j

        The lower bound of alpha_j, L, is::
            L = max(0, alpha_i + alpha_j - C), if y_i == y_j
                max(0, alpha_j - alpha_i),     if y_i != y_j

        The clipping rule for alpha_j is::

                                              H, if H <= alpha_j (unclipped)
            alpha_j (new) = alpha_j (unclipped), if L <  alpha_j (unclipped) <  H
                                              L, if      alpha_j (unclipped) <= L

        :param unclipped_alpha2: The unclipped value of alpha_j
        :return: The clipped value of alpha_j
        """
        if self._y[i][0] == self._y[j][0]:
            H = min(self.C, self._alpha[i] + self._alpha[j])
            L = max(0, self._alpha[i] + self._alpha[j] - self.C)
        else:
            H = min(self.C, self._alpha[j] - self._alpha[i] + self.C)
            L = max(0, self._alpha[j] - self._alpha[i])

        if unclipped_alpha2 >= H:
            return H, L, H
        if unclipped_alpha2 <= L:
            return L, L, H
        return unclipped_alpha2, L, H

    def _get_new_b(self, i, j, new_alpha1, new_alpha2, L, H):
        """
        Get the new intercept according to new_alpha_i and new_alpha_j. The formulas are too complex and are
        omitted here.
        """
        if 0 < new_alpha1 < self.C:
            return (-self._E[i] - self._y[i][0] * self._X_dot[i][i] * (new_alpha1 - self._alpha[i]) -
                    self._y[j][0] * self._X_dot[i][j] * (new_alpha2 - self._alpha[j]) + self._b)
        if 0 < new_alpha2 < self.C:
            return (-self._E[j] - self._y[i][0] * self._X_dot[i][j] * (new_alpha1 - self._alpha[i]) -
                    self._y[j][0] * self._X_dot[j][j] * (new_alpha2 - self._alpha[j]) + self._b)
        if L != H:
            b1 = (-self._E[i] - self._y[i][0] * self._X_dot[i][i] * (new_alpha1 - self._alpha[i]) -
                  self._y[j][0] * self._X_dot[i][j] * (new_alpha2 - self._alpha[j]) + self._b)
            b2 = (-self._E[j] - self._y[i][0] * self._X_dot[i][j] * (new_alpha1 - self._alpha[i]) -
                  self._y[j][0] * self._X_dot[j][j] * (new_alpha2 - self._alpha[j]) + self._b)
            return 0.5 * (b1 + b2)
        return self._b

    def update_E(self, i, j, new_alpha1, new_alpha2, new_b):
        delta_b = new_b - self._b
        for k, old_E in enumerate(self._E):
            self._E[k] = (old_E +
                          self._y[i][0] * self._X_dot[i][k] * (new_alpha1 - self._alpha[i]) +
                          self._y[j][0] * self._X_dot[j][k] * (new_alpha2 - self._alpha[j]) +
                          delta_b)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept


class Perceptron(ClassificationModels):

    def __init__(self, pre_process=False, lr=0.01):
        super().__init__(pre_process=pre_process, fill_one=False)
        self.lr = lr

    def fit(self, X, y):
        self._preprocess(X, y)

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.pre_process:
            self._y = np.zeros_like(self.y)
            self._class_num = self._preprocess_y()
        else:
            self._y = self.y
            self._class_num = len(np.unique(self._y))

        if self._class_num == 1:
            warnings.warn('There is only one group of y')
        elif self._class_num == 2:
            self._coef = np.zeros(self._X_shape[1])
            self._intercept = 0
        else:
            raise NotImplementedError("Perceptron doesn't support multiple classes of y.")


class NeuralNetwork(SupervisedModels):

    def __init__(self):
        super().__init__()
