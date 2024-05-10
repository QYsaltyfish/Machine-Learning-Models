import numpy as np
from . import models
from . import metrics


def cross_val_score(model: models.SupervisedModels, X, y, cv=5, scoring='mean_squared_error', random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    num_samples = X.shape[0]
    fold_size = num_samples // cv
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    if scoring == 'mean_squared_error':
        f_score = metrics.mean_squared_error
    else:
        raise NotImplementedError(f'scoring {scoring} is not implemented')

    scores = np.zeros(cv)

    for i in range(cv):
        val_indices = indices[fold_size * i: fold_size * (i + 1)]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        model.fit(X_train, y_train)
        scores[i] = f_score(y_val, model.predict(X_val))

    return scores
