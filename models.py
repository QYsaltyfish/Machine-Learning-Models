import numpy as np
import cupy as cp
import copy
import warnings


class Models:

    def __init__(self, gpu=False):
        self._X_shape = None
        if gpu:
            self._p = cp
        else:
            self._p = np
        self.X_dicts = None
        self._X_dtypes = None

    def predict(self, X):
        raise Exception("This is an empty model.")

    def _describe(self):
        self._X_shape = self._X.shape

    def _preprocess(self, X, y):
        try:
            X = copy.deepcopy(X)
            X = self._preprocess_categorical_X(X)

            self._X = self._p.array(X)
            self._X_dtypes = np.zeros(self._X.shape[1], dtype=np.int8)
            for j, column in enumerate(self._X.T):
                if len(np.unique(column)) < column.shape[0] // 2:
                    self._X_dtypes[j] = 0
                else:
                    self._X_dtypes[j] = 1

        except Exception:
            raise TypeError("X is not or cannot be converted into a ndarray.")

    def _preprocess_categorical_X(self, X):
        first_row = X[0]
        self.X_dicts: list[None or dict] = [None for _ in range(len(first_row))]

        for j, elem in enumerate(first_row):
            if isinstance(elem, str):
                dic = dict()
                for i in range(len(X)):
                    if X[i][j] in dic:
                        X[i][j] = dic[X[i][j]]
                    else:
                        dic[X[i][j]] = X[i][j] = len(dic)
                self.X_dicts[j] = dic
        return X

    def _preprocess_X_test(self, X):
        if self.X_dicts is None:
            return self._p.array(X)

        X = copy.deepcopy(X)
        for j, dic in enumerate(self.X_dicts):
            if dic is not None:
                for i in range(len(X)):
                    if X[i][j] in dic:
                        X[i][j] = dic[X[i][j]]
                    else:
                        warnings.warn(f"{X[i][j]} is not in the training data and is automatically set to 0.")
                        X[i][j] = 0
        return self._p.array(X)

    @property
    def X_shape_(self):
        return self._X_shape


class SupervisedModels(Models):

    def __init__(self, gpu=False):
        super().__init__(gpu=gpu)

        self._y_shape = None
        self.y_dict = None

    def _describe(self):
        super()._describe()
        self._y_shape = self._y.shape

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        try:
            y = y.copy()
            self._preprocess_categorical_y(y)
            self._y = self._p.array(y)
            if len(np.unique(y)) < self._y.shape[0] // 2:
                self._y_dtype = 0
            else:
                self._y_dtype = 1
        except Exception:
            raise TypeError("y is not or cannot be converted into a ndarray.")

        self._describe()

    def _preprocess_categorical_y(self, y):
        if isinstance(y[0], str):
            self.y_dict = dict()
            for i, elem in enumerate(y):
                if elem not in self.y_dict:
                    self.y_dict[elem] = y[i] = len(self.y_dict)
                else:
                    y[i] = self.y_dict[elem]
        return y

    @property
    def y_shape_(self):
        return self._y_shape

    def fit(self, X, y):
        raise Exception("This is an empty model.")


class PredictionModels(SupervisedModels):

    def __init__(self, add_constant=False, gpu=False):
        super().__init__(gpu=gpu)
        self.add_constant = add_constant
        self._SST = None
        self._SSR = None
        self._SSE = None
        self.R_squared = None
        self._X_test = None
        self._y_predict = None
        self._y_fit = None

    @property
    def sst_(self):
        return self._SST

    @property
    def sse_(self):
        return self._SSE

    @property
    def ssr_(self):
        return self._SSR

    def _preprocess(self, X, y):
        super()._preprocess(X, y)
        if self.add_constant:
            self._X = self._p.hstack((self._p.ones((self._X_shape[0], 1)), self._X))
            self._X_shape = self._X.shape

        _y_mean = np.mean(y)
        self._SST = self._p.sum((self._y - _y_mean) ** 2)


class LinearRegression(PredictionModels):

    def __init__(self, add_constant=True, penalty='l2', C=0, max_iter=100, tol=1e-5, gpu=False):
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

        :param add_constant: Whether constants should be added to X
        :param penalty: The type of regularization
        :param C: Coefficient of regularization
        :param max_iter: Max iterating times for Lasso regression
        :param tol: Tolerance for convergence criteria
        :param gpu: Whether to use GPU or not
        """

        super().__init__(add_constant=add_constant, gpu=gpu)

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
            self._coef = self._p.linalg.inv(self._X_T_X + self.C * self._p.eye(self._X_shape[1])) @ self._X_T_y
        else:
            is_converge = False
            self._coef = self._p.zeros(self._X_shape[1])
            for _ in range(self.max_iter):
                start_coef = self._coef.copy()
                for k in range(self._X_shape[1]):
                    best_coef_k = self._best_coef_k(k)
                    self._coef[k] = best_coef_k
                if self._p.max(self._coef - start_coef) < self.tol:
                    is_converge = True
                    break
            if not is_converge:
                warnings.warn("The model doesn't converge")

        self._y_fit = self._X @ self._coef
        self._SSR = self._p.sum((self._y_fit - self._y) ** 2)
        self._SSE = self._SST - self._SSR
        self.R_squared = self._SSR / self._SST

    def predict(self, X_test):
        if self._coef is None:
            raise Exception("This model has not fitted yet.")

        X_test = self._p.array(X_test)

        if self.add_constant:
            X_test = self._p.hstack((self._p.ones((X_test.shape[0], 1)), X_test))

        return X_test @ self._coef

    def _preprocess(self, X, y):
        super()._preprocess(X, y)
        self._X_T = self._X.T
        self._X_T_X = self._X_T @ self._X
        self._X_T_y = self._X_T @ self._y

    def _best_coef_k(self, k):
        return self._sub_gradient(self._X_T_y[k] - self._X_T_X[k] @ self._coef +
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

    def __init__(self, pre_process=False, add_constant=False, gpu=False):
        super().__init__(gpu=gpu)
        self._y = None
        self.pre_process = pre_process
        self.add_constant = add_constant
        self._class_num = None

    def _preprocess_y(self):
        y_dict = dict()
        for i, y in enumerate(self._y):
            if y not in y_dict:
                self._y[i] = len(y_dict)
                y_dict[y] = len(y_dict)
            else:
                self._y[i] = y_dict[y]
        return len(y_dict)

    @property
    def class_num_(self):
        return self._class_num

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.add_constant:
            self._X = self._p.hstack((self._p.ones((self._X_shape[0], 1)), self._X))


class LogisticRegression(ClassificationModels):

    def __init__(self, add_constant=True, pre_process=False, lr=0.04, penalty='l2', C=0, max_iter=10000, tol=1e-5,
                 batch='all', gpu=False):
        """
        Initialize the linear regression model with given parameters.

        The objective function for l2 regularization is::

            l(w) + C / 2 * ||w||_2^2

        The optimization is done by gradient descent.

        :param add_constant: Whether constants should be added to X
        :param pre_process: Whether to preprocess vector y
        :param lr: The learning rate for gradient descent
        :param penalty: The type of regularization
        :param C: Coefficient of regularization
        :param max_iter: Max iterating times for Lasso regression
        :param tol: Tolerance for convergence criteria
        :param batch: The batch size for gradient descent
        """
        super().__init__(pre_process=pre_process, add_constant=add_constant, gpu=gpu)

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
        if batch == 'all':
            self.batch = self._p.inf
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
            self._class_num = self._preprocess_y()
        else:
            self._class_num = len(self._p.unique(self._y))

        if self._class_num == 1:
            raise Exception('There is only one group of y')
        elif self._class_num == 2:
            self._coef = self._p.zeros(self._X_shape[1] + self.add_constant)
        else:
            self._coef = self._p.zeros((self._class_num - 1, self._X_shape[1] + self.add_constant))

        if self.batch >= self._X_shape[0]:
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
            if self._p.max(abs(curr_coef - self._coef)) < self.tol:
                break

    def _binary_gradient_descent(self):
        index_list = self._p.random.choice(self._p.arange(self._X_shape[0]), self._batch, replace=False)
        self._coef += self.lr * (sum(
            self._X[i] * (self._y[i] - self._sigmoid(self._p.dot(self._X[i], self._coef)))
            for i in index_list
        ) - self.C * self._coef)

    def _multiclass_solver(self):
        # TODO
        pass

    def predict(self, X, threshold=0.5, add_constant=True):
        X = self._preprocess_X_test(X)
        if self._class_num == 2:
            res = self.predict_prob(X, add_constant=add_constant)
            for i in range(len(res)):
                res[i] = 0 if res[i] < threshold else 1
            return res
        else:
            # TODO
            pass

    def predict_prob(self, X, add_constant=True):
        if self._class_num == 2:
            res = [0 for _ in range(X.shape[0])]

            if add_constant:
                X = self._p.hstack((self._p.ones((X.shape[0], 1)), X))

            for i, x in enumerate(X):
                res[i] = self._sigmoid(self._p.dot(x, self._coef))
            return res
        else:
            # TODO
            pass

    @property
    def coef_(self):
        return self._coef

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


class SVM(ClassificationModels):

    def __init__(self, pre_process=True, margin='soft', C=1, eps=1e-3, gpu=False):
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
        super().__init__(gpu=gpu)
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

        if any(self._y == 0):
            for i, y in enumerate(self._y):
                if y == 0:
                    self._y[i] = -1
            return

        if any(self._y == -1):
            for i, y in enumerate(self._y):
                if y != -1:
                    self._y[i] = 1
            return

        for i, y in enumerate(self._y):
            if y not in y_dict:
                if len(y_dict) == 0:
                    y_dict[y] = -1
                else:
                    y_dict[y] = 1
            self._y[i] = y_dict[y]

    def fit(self, X, y):
        self._preprocess(X, y)

        if self._margin == 0:
            self._hard_smo_solver()
        elif self._margin == 1:
            self._soft_smo_solver()

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.pre_process:
            self._preprocess_y()
            self._class_num = 2
        else:
            self._class_num = len(self._p.unique(self._y))

        if self._class_num == 1:
            raise Exception('There is only one group of y')
        elif self._class_num == 2:
            self._coef = self._p.zeros(self._X_shape[1])
            self._intercept = 0
        else:
            raise NotImplementedError("SVM doesn't support multiple classes of y.")

        self._X_dot = self._p.zeros((self._X_shape[0], self._X_shape[0]))
        self._d = self._p.zeros(self._X_shape[0])
        self._E = self._p.zeros(self._X_shape[0])
        self._alpha = self._p.zeros(self._X_shape[0])

        for i in range(self._X_shape[0]):
            for j in range(self._X_shape[0]):
                self._X_dot[i][j] = self._p.dot(self._X[i], self._X[j])

        for i in range(self._X_shape[0]):
            self._E[i] = -self._y[i]

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
                    j = self._p.random.choice(l[:i] + l[i + 1:])
                    # j = self._find_second_alpha(i)
                else:
                    continue
                new_alpha2, L, H = self._clip_alpha2(i, j, self._get_unclipped_alpha2(i, j))
                new_alpha1 = self._alpha[i] + self._y[i] * self._y[j] * (self._alpha[j] - new_alpha2)
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
            non_bound_indexes = []
            for i in range(self._X_shape[0]):
                if 0 < self._alpha[i] < self.C:
                    non_bound_indexes.append(i)

            # Second traversal, traverse all the non-bound samples and optimize, until all non-bound samples
            # satisfy KKT conditions.
            while True:
                find_violator_this_loop = False
                for i in non_bound_indexes:
                    if self._check_epsilon_kkt_violation(i):
                        find_violator_this_loop = True
                        find_violator = True
                        l = list(range(self._X_shape[0]))
                        j = self._p.random.choice(l[:i] + l[i + 1:])
                        # j = self._find_second_alpha(i)
                    else:
                        continue
                    new_alpha2, L, H = self._clip_alpha2(i, j, self._get_unclipped_alpha2(i, j))
                    new_alpha1 = self._alpha[i] + self._y[i] * self._y[j] * (self._alpha[j] - new_alpha2)
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

        self._coef = sum((self._alpha[i] * self._y[i] * self._X[i] for i in range(self._X_shape[0])))
        b_star = [self._y[j] -
                  sum(self._alpha[i] * self._y[i] * self._X_dot[i][j]
                      for i in range(self._X_shape[0]) if self._alpha[i] != 0)
                  for j in range(self._X_shape[0]) if 0 < self._alpha[j] < self.C]
        self._intercept = self._p.mean(b_star)

    def _check_epsilon_kkt_violation(self, i):
        """
        Check the KKT conditions within precision epsilon.

        If     alpha_i = 0, then 1 - eps <= y_i * d_i
        If 0 < alpha_i < C, then 1 - eps <= y_i * d_i <= 1 + eps
        If     alpha_i = C, then            y_i * d_i <= 1 + eps

        :param i: The index of the coefficient to check KKT condition.
        :return: Whether the coefficient **violates** epsilon-KKT condition.
        """
        return (self._alpha[i] < self.C and self._y[i] * self._E[i] < -self.eps
                or
                self._alpha[i] > 0 and self._y[i] * self._E[i] > self.eps)

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
        Get the unclipped value of alpha_j by optimizing both alpha_i and alpha_j in the dual problem.

        The update rule is::

            alpha_j (unclipped) = alpha_j (old) + y_2 * (E_i - E_j) / (k_ii + k_jj - 2k_ij)

        :param i: The index of the first alpha
        :param j: The index of the second alpha
        :return: Unclipped alpha_j
        """
        eta = self._X_dot[i][i] + self._X_dot[j][j] - 2 * self._X_dot[i][j]
        if eta > 0:
            return self._alpha[j] + self._y[j] * (self._E[i] - self._E[j]) / eta
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
        if self._y[i] == self._y[j]:
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
            return (-self._E[i] - self._y[i] * self._X_dot[i][i] * (new_alpha1 - self._alpha[i]) -
                    self._y[j] * self._X_dot[i][j] * (new_alpha2 - self._alpha[j]) + self._b)
        if 0 < new_alpha2 < self.C:
            return (-self._E[j] - self._y[i] * self._X_dot[i][j] * (new_alpha1 - self._alpha[i]) -
                    self._y[j] * self._X_dot[j][j] * (new_alpha2 - self._alpha[j]) + self._b)
        if L != H:
            b1 = (-self._E[i] - self._y[i] * self._X_dot[i][i] * (new_alpha1 - self._alpha[i]) -
                  self._y[j] * self._X_dot[i][j] * (new_alpha2 - self._alpha[j]) + self._b)
            b2 = (-self._E[j] - self._y[i] * self._X_dot[i][j] * (new_alpha1 - self._alpha[i]) -
                  self._y[j] * self._X_dot[j][j] * (new_alpha2 - self._alpha[j]) + self._b)
            return 0.5 * (b1 + b2)
        return self._b

    def update_E(self, i, j, new_alpha1, new_alpha2, new_b):
        delta_b = new_b - self._b
        for k, old_E in enumerate(self._E):
            self._E[k] = (old_E +
                          self._y[i] * self._X_dot[i][k] * (new_alpha1 - self._alpha[i]) +
                          self._y[j] * self._X_dot[j][k] * (new_alpha2 - self._alpha[j]) +
                          delta_b)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept


class Perceptron(ClassificationModels):

    def __init__(self, pre_process=False, lr=0.01):
        super().__init__(pre_process=pre_process, add_constant=False)
        self.lr = lr

    def fit(self, X, y):
        self._preprocess(X, y)

        for x, y in zip(self._X, self._y):
            y_pred = self._get_y_pred(x)

            if y == y_pred:
                continue

            self._coef += self.lr * (y - y_pred) * x
            self._intercept += self.lr * (y - y_pred)

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        if self.pre_process:
            self._class_num = self._preprocess_y()
        else:
            self._class_num = len(self._p.unique(self._y))

        if self._class_num == 1:
            warnings.warn('There is only one group of y')
        elif self._class_num == 2:
            self._coef = self._p.zeros(self._X_shape[1])
            self._intercept = 0
        else:
            raise NotImplementedError("Perceptron doesn't support multiple classes of y.")

    def _get_y_pred(self, x):
        if self._coef @ x + self._intercept >= 0:
            return 1
        else:
            return 0

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept


class DecisionTree(ClassificationModels):
    class DecisionTreeNode:

        def __init__(self, X=None, y=None, is_end=False, outcome=None):
            self.X = X
            self.y = y
            self.children = None
            self.attrib = None
            self.value = None
            self.is_end = is_end
            self.outcome = outcome
            self.entropy = None

        def clear(self):
            self.X = None
            self.y = None
            self.entropy = None

        def get_outcome(self, y_uniques):
            if self.outcome == -1:
                return np.random.choice(y_uniques)
            return self.outcome

    class CARTTreeNode:

        def __init__(self, X=None, y=None, is_end=False, outcome=None):
            self.X = X
            self.y = y
            self.left = None
            self.right = None
            self.attrib = None
            self.value = None
            self.is_end = is_end
            self.outcome = outcome

        def clear(self):
            self.X = None
            self.y = None

        def get_outcome(self, y_uniques):
            if self.outcome == -1:
                return np.random.choice(y_uniques)
            return self.outcome

    def __init__(self, pre_process=False, solver='ID3', gini_threshold=0.05):
        super().__init__(pre_process=pre_process)

        if solver == 'ID3':
            self.solver = self._ID3_solver
            self.predictor = self._ID3_predictor
        elif solver == 'C4.5':
            self.solver = self._C45_solver
            self.predictor = self._C45_predictor
        elif solver == 'CART':
            self.solver = self._CART_solver
            self.predictor = self._CART_predictor
            self.gini_threshold = gini_threshold
        else:
            raise NotImplementedError("Decision tree solver doesn't support " + solver)
        self.root = None

    def fit(self, X, y):
        self._preprocess(X, y)
        self.solver()

    def predict(self, X):
        return self.predictor(X)

    def _preprocess(self, X, y):
        super()._preprocess(X, y)

        super()._preprocess_y()

        self._X_uniques = [np.unique(X) if self._X_dtypes[j] == 0 else None for j, X in enumerate(self._X.T)]
        self._y_uniques = np.unique(self._y)
        self._X_sorts = [sorted(list(set(X))) if self._X_dtypes[j] == 1 else None for j, X in enumerate(self._X.T)]
        self._X_sorts = [[(sorted_X[i] + sorted_X[i + 1]) / 2
                          for i in range(len(sorted_X) - 1)] if sorted_X is not None else None
                         for sorted_X in self._X_sorts]

    def _ID3_predictor(self, X):
        X = self._preprocess_X_test(X)
        res = [None for _ in range(X.shape[0])]
        for i, x in enumerate(X):
            node = self.root
            while not node.is_end:
                node = node.children[x[node.attrib]]
            res[i] = node.get_outcome(self._y_uniques)
        return res

    def _ID3_solver(self):
        self.root = self.DecisionTreeNode(self._X, self._y)
        self._ID3_recursive(self.root)

    def _ID3_recursive(self, node):
        if node.is_end:
            return

        node.entropy = self._entropy(node.y)

        if node.entropy == 0:
            node.is_end = True
            node.outcome = node.y[0]
            node.clear()
            return

        best_attrib, best_gain = max(((attrib2, gain)
                                      for attrib2, gain in enumerate(
            self._information_gain(node, attrib1) for attrib1 in range(self._X_shape[1])
        )), key=lambda x: x[1])

        node.attrib = best_attrib

        index_masks = [node.X[:, best_attrib] == X_value for X_value in self._X_uniques[best_attrib]]
        node.children = [self.DecisionTreeNode(X=node.X[mask], y=node.y[mask])
                         if np.sum(mask) != 0
                         else self.DecisionTreeNode(is_end=True, outcome=-1)
                         for mask in index_masks]

        node.clear()
        for child in node.children:
            self._ID3_recursive(child)
        return

    def _C45_predictor(self, X):
        X = self._preprocess_X_test(X)
        res = [None for _ in range(X.shape[0])]
        for i, x in enumerate(X):
            node = self.root
            while not node.is_end:
                if self._X_dtypes[node.attrib] == 0:
                    node = node.children[x[node.attrib]]
                else:
                    if x[node.attrib] <= node.value:
                        node = node.children[0]
                    else:
                        node = node.children[1]
            res[i] = node.get_outcome(self._y_uniques)
        return res

    def _C45_solver(self):
        self.root = self.DecisionTreeNode(self._X, self._y)
        self._C45_recursive(self.root)

    def _C45_recursive(self, node):
        if node.is_end:
            return

        node.entropy = self._entropy(node.y)

        if node.entropy == 0:
            node.is_end = True
            node.outcome = node.y[0]
            node.clear()
            return

        best_attrib, best_ratio, best_value = (
            max(((attrib, self._information_gain_ratio(node, attrib), None)
                 if self._X_dtypes[attrib] == 0
                 else (attrib,) + max(((self._information_gain_ratio(node, attrib, value), value)
                                       if any(value < node.X[:, attrib]) and any(value > node.X[:, attrib])
                                       else (0, 0)
                                       for value in self._X_sorts[attrib]),
                                      key=lambda x: x[0])
                 for attrib in range(self._X_shape[1])),
                key=lambda x: x[1]))

        node.attrib = best_attrib
        node.value = best_value

        if self._X_dtypes[best_attrib] == 0:
            index_masks = [node.X[:, best_attrib] == X_value for X_value in self._X_uniques[best_attrib]]
        else:
            index_masks = [node.X[:, best_attrib] <= best_value, None]
            index_masks[1] = ~index_masks[0]

        node.children = [self.DecisionTreeNode(X=node.X[mask], y=node.y[mask])
                         if np.sum(mask) != 0
                         else self.DecisionTreeNode(is_end=True, outcome=-1)
                         for mask in index_masks]

        node.clear()
        for child in node.children:
            self._C45_recursive(child)
        return

    def _CART_predictor(self, X):
        X = self._preprocess_X_test(X)
        res = [None for _ in range(X.shape[0])]
        for i, x in enumerate(X):
            node = self.root
            while not node.is_end:
                if self._X_dtypes[node.attrib] == 0:
                    if x[node.attrib] == node.value:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if x[node.attrib] < node.value:
                        node = node.left
                    else:
                        node = node.right
            res[i] = node.get_outcome(self._y_uniques)
        return res

    def _CART_solver(self):
        self.root = self.CARTTreeNode(self._X, self._y)
        if self._y_dtype == 0:
            self._CART_recursive_discrete(self.root)
        else:
            raise NotImplementedError('CART solver only implemented for discrete variables')

    def _CART_recursive_discrete(self, node):
        if node.is_end:
            return

        if self._gini(node.y) == 0:
            node.is_end = True
            node.outcome = node.y[0]
            node.clear()
            return

        best_attrib, best_gini, best_value = (
            min(((attrib,) + min((self._conditional_gini(node, attrib, value), value)
                                 for value in self._X_uniques[attrib])
                 if self._X_dtypes[attrib] == 0
                 else (attrib,) + min(((self._conditional_gini(node, attrib, value), value)
                                       if any(value < node.X[:, attrib]) and any(value > node.X[:, attrib])
                                       else (np.inf, 0)
                                       for value in self._X_sorts[attrib]),
                                      key=lambda x: x[0])
                 for attrib in range(self._X_shape[1])),
                key=lambda x: x[1]))

        node.attrib = best_attrib
        node.value = best_value

        if self._X_dtypes[best_attrib] == 0:
            index_mask = node.X[:, best_attrib] == best_value
        else:
            index_mask = node.X[:, best_attrib] < best_value
        not_index_mask = ~index_mask

        if best_gini < self.gini_threshold:
            node.left = self.CARTTreeNode(is_end=True, outcome=self._mode(node.y[index_mask]))
            node.right = self.CARTTreeNode(is_end=True, outcome=self._mode(node.y[not_index_mask]))
            node.clear()
            return

        if np.sum(index_mask) != 0:
            node.left = self.CARTTreeNode(X=node.X[index_mask], y=node.y[index_mask])
        else:
            node.left = self.CARTTreeNode(is_end=True, outcome=-1)

        if np.sum(not_index_mask) != 0:
            node.right = self.CARTTreeNode(X=node.X[not_index_mask], y=node.y[not_index_mask])
        else:
            node.right = self.CARTTreeNode(is_end=True, outcome=-1)

        node.clear()
        if not node.left.is_end:
            self._CART_recursive_discrete(node.left)
        if not node.right.is_end:
            self._CART_recursive_discrete(node.right)

    def _information_gain(self, node, attrib):
        return node.entropy - self._conditional_entropy_ID3(node.X, node.y, attrib)

    def _entropy(self, y):
        return -np.sum(
            ratio * np.log2(ratio)
            for ratio in [
                np.sum(y == y_value) / len(y)
                for y_value in self._y_uniques
            ]
            if ratio > 0
        )

    def _conditional_entropy_ID3(self, X, y, attrib):
        return np.sum(
            self._entropy(sub_y) * len(sub_y) / len(y) if len(sub_y) != 0 else 0
            for sub_y in [
                y[index_mask]
                for index_mask in [
                    X[:, attrib] == X_value
                    for X_value in self._X_uniques[attrib]
                ]
            ]
        )

    def _information_gain_ratio(self, node, attrib, value=None):
        if self._X_dtypes[attrib] == 0:
            ys = [node.y[index_mask]
                  for index_mask
                  in [node.X[:, attrib] == X_value for X_value in self._X_uniques[attrib]]]
        else:
            index_mask = node.X[:, attrib] <= value
            ys = [node.y[index_mask], node.y[~index_mask]]

        return (node.entropy - self._conditional_entropy_C45(ys, len(node.y))) / self._intrinsic_value(ys, len(node.y))

    def _conditional_entropy_C45(self, ys, y_total):
        return np.sum(
            self._entropy(y) * len(y) / y_total if len(y) != 0 else 0
            for y in ys
        )

    @staticmethod
    def _intrinsic_value(ys, y_total):
        ratios = [len(y) / y_total for y in ys]
        res = -np.sum(ratio * np.log2(ratio) for ratio in ratios if ratio != 0)
        if res == 0:
            print(ys, y_total)
        return res

    def _gini(self, y):
        return 1 - np.sum(
            ratio ** 2
            for ratio in [
                np.sum(y == y_value) / len(y)
                for y_value in self._y_uniques
            ]
        )

    def _conditional_gini(self, node, attrib, value):
        if self._X_dtypes[attrib] == 0:
            mask = node.X[:, attrib] == value
        else:
            mask = node.X[:, attrib] < value
        y1 = node.y[mask]
        y2 = node.y[~mask]
        return np.sum([len(y) / len(node.y) * self._gini(y) for y in (y1, y2) if len(y) > 0])

    @staticmethod
    def _mode(y):
        dic = dict()
        for y_value in y:
            dic[y_value] = dic.get(y_value, 0) + 1
        return max(dic, key=dic.get)


class NeuralNetwork(SupervisedModels):
    class Node:

        def __init__(self, activation_function=None, **kwargs):
            if activation_function == 'constant':
                self.f = self._constant
                self.df = self._d_constant
                self._name = 'constant'
            elif activation_function == 'linear':
                self.f = self._linear
                self.df = self._d_linear
                self.k = kwargs.get('k', 1)
                self._name = f'k={self.k} linear'
            elif activation_function == 'sigmoid':
                self.f = self._sigmoid
                self.df = self._d_sigmoid
                self._name = 'sigmoid'
            elif activation_function == 'tanh':
                self.f = self._tanh
                self.df = self._d_tanh
                self._name = 'tanh'
            elif activation_function == 'relu':
                self.f = self._relu
                self.df = self._d_relu
                self._name = 'relu'
            elif activation_function == 'p-relu':
                self.f = self._p_relu
                self.df = self._d_p_relu
                self.p = kwargs.get('p', 0.01)
                self._name = 'p-relu'

            self.linear_comb = None
            self.output = None

        def __repr__(self):
            return f'{self._name} node'

        def _sigmoid(self):
            self.output = 1 / (1 + np.exp(-self.linear_comb))
            return self.output

        def _d_sigmoid(self):
            return self.output * (1 - self.output)

        def _linear(self):
            self.output = self.linear_comb * self.k
            return self.output

        def _d_linear(self):
            return self.k

        def _constant(self):
            self.output = 1
            return 1

        @staticmethod
        def _d_constant():
            return 0

        def _tanh(self):
            self.output = np.tanh(self.linear_comb)
            return self.output

        def _d_tanh(self):
            return 1 - self.output ** 2

        def _relu(self):
            self.output = max(self.linear_comb, 0)
            return self.output

        def _d_relu(self):
            if self.linear_comb >= 0:
                return 1
            return 0

        def _p_relu(self):
            if self.linear_comb >= 0:
                self.output = self.linear_comb
            else:
                self.output = self.p * self.linear_comb
            return self.output

        def _d_p_relu(self):
            if self.linear_comb >= 0:
                return 1
            return self.p

    def __init__(self, lr=0.01, max_iter=1, weight_init='xavier', gpu=False):
        super().__init__(gpu=gpu)

        self.layers: list[list[NeuralNetwork.Node]] = [[], []]
        self.weights = None
        self.lr = lr
        self.df = self._p.vectorize(lambda x: x.df())
        self.max_iter = max_iter

        if weight_init == 'xavier':
            self.weight_init = 1
        elif weight_init == 'random':
            self.weight_init = 2
        else:
            self.weight_init = 0

    def set_input_layer(self, input_length, add_constant=False):
        self.layers[0] = [self.Node() for _ in range(input_length)]

        if add_constant:
            self.layers[0].append(self.Node('constant'))
            raise NotImplementedError("Adding constant hasn't been implemented yet.")

    def set_output_layer(self, output_length, activation_function=None):
        if activation_function is None:
            raise Exception('An activation function must be specified.')

        self.layers[-1] = [self.Node(activation_function=activation_function) for _ in range(output_length)]

    def add_hidden_layer(self, index=-1, hidden_length=0, activation_function=None, add_constant=False):
        if activation_function is None:
            warnings.warn("No activation function is specified.")

        if hidden_length == 0:
            raise Exception("Hidden length must be a positive integer.")

        self.layers.insert(index, [self.Node(activation_function=activation_function) for _ in range(hidden_length)])

        if add_constant:
            raise NotImplementedError("Adding constant hasn't been implemented yet.")

    def _generate_weights(self):
        if self.weight_init == 0:
            self._generate_zero_weights()
        elif self.weight_init == 1:
            self._generate_xavier_weights()
        elif self.weight_init == 2:
            self._generate_random_weights()

    def _generate_zero_weights(self):
        self.weights = [self._p.zeros((len(self.layers[i]), len(self.layers[i + 1])))
                        for i in range(len(self.layers) - 1)]

    def _generate_random_weights(self):
        self.weights = [self._p.random.random((len(self.layers[i]), len(self.layers[i + 1])))
                        for i in range(len(self.layers) - 1)]

    def _generate_xavier_weights(self):
        self.weights = [None for _ in range(len(self.layers) - 1)]
        for layer_index in range(len(self.layers) - 1):
            len_in = len(self.layers[layer_index])
            len_out = len(self.layers[layer_index + 1])
            bound = self._p.sqrt(6 / (len_in + len_out))
            self.weights[layer_index] = self._p.random.uniform(-bound, bound, size=(len_in, len_out))

    def _forward_propagate(self, x):
        curr_layer = x  # curr_layer is a reference of x

        first_layer = self.layers[0]
        for i, x_i in enumerate(x):
            first_layer[i].linear_comb = first_layer[i].output = x_i

        for i in range(len(self.layers) - 1):
            curr_layer = curr_layer @ self.weights[i]
            for j, node in enumerate(self.layers[i + 1]):
                node.linear_comb = curr_layer[j]
                curr_layer[j] = node.f()

    def _backward_propagate(self, y_real):
        delta_weights = [None for _ in range(len(self.weights))]

        # The last layer is special
        grad = self._p.array([(node.output - y_real[i]) * node.df() for i, node in enumerate(self.layers[-1])])

        if all(grad) == 0:
            return

        last_act = self._p.array([node.output for node in self.layers[-2]])
        delta_weights[-1] = self._p.outer(last_act, grad)

        # Process remaining layers
        for layer_index in range(-2, -len(self.layers), -1):
            grad = self.weights[layer_index + 1] @ grad
            grad = self.df(self.layers[layer_index]) * grad
            last_act = self._p.array([node.output for node in self.layers[layer_index - 1]])
            delta_weights[layer_index] = self._p.outer(last_act, grad)

        # Update weights
        for i, delta_weight in enumerate(delta_weights):
            self.weights[i] -= delta_weight

    def fit(self, X, y):
        y = self._preprocess_y(y)
        if self.weights is None:
            self._generate_weights()

        for _ in range(self.max_iter):
            for x, y_real in zip(X, y):
                self._forward_propagate(x)
                self._backward_propagate(y_real)

    def predict(self, X):
        res = self._p.zeros((len(X), len(self.layers[-1])))
        for i, x in enumerate(X):
            self._forward_propagate(x)
            for j, node in enumerate(self.layers[-1]):
                res[i][j] = node.output
        return res

    @staticmethod
    def _preprocess_y(y):
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return y
