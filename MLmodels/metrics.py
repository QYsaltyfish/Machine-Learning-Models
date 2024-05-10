import numpy as np


def _convert_array(*args):
    return [np.array(arg) for arg in args]


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = _convert_array(y_true, y_pred)
    return np.sum((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _convert_array(y_true, y_pred)
    return np.sum(np.abs(y_true - y_pred))


def neg_mean_squared_error(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)

