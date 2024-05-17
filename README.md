# Machine Learning Models from Scratch in Python

## Overview

This repository contains implementations of various machine learning models from scratch in Python.

## Models Included

1. **Multiple Linear Regression**: Including Lasso & Ridge regularization techniques.
2. **Logistic Regression**: Implementations of both gradient descent and newton solver.
3. **Support Vector Machine (SVM)**: Implementations of both hard-margin and soft-margin SVM.
4. **Neural Network**: Implementations of both Adam and SGD optimizer.
5. **Decision Tree**: ID3 & C4.5 (post-pruning to be implemented) & CART algorithms.
6. **K-means**

## Code Usage Examples

Here are some code snippets to demonstrate how to use the models:

### Multiple Linear Regression

```python
from MLmodels.models import LinearRegression

# Create a model instance
lr = LinearRegression()

# Fit the model
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)
```

#### Regularization Options

The `LinearRegression` model supports regularization through the `penalty` and `C` parameters.

* To use Lasso regression (L1 regularization), set `penalty='l1'`.

* To use Ridge regression (L2 regularization), set `penalty='l2'`.

The parameter `C` controls the strength of the regularization; larger values of `C` specify stronger regularization.

```python
# Create a model instance with L1 regularization (Lasso)
lr_l1 = LinearRegression(penalty='l1', C=1)

# Create a model instance with L2 regularization (Ridge)
lr_l2 = LinearRegression(penalty='l2', C=1)
```

### Logistic Regression

```python
from MLmodels.models import LogisticRegression

# Create a model instance
lr = LogisticRegression()

# Fit the model
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)
```

#### Solver Options

The `Logistic Regression` model supports two solvers: Newton's method (default) and gradient descent.

To use gradient descent as the solver, you can specify it during the model initialization:

```python
# Create a model instance using gradient descent as the solver
lr = LogisticRegression(solver='gradient descent')
```

#### Regularization

You can adjust the regularization strength using the parameter `C`. The parameter `C` is the regularization strength; larger values specify stronger regularization.

```python
# Create a model instance with regularization parameter C
lr = LogisticRegression(C=1)
```

### SVM

```python
from MLmodels.models import SVM

# Create a model instance
svm = SVM()

# Fit the model
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)
```

### Neural Network

```python
from MLmodels.models import NeuralNetwork

# Create a model instance
nn = NeuralNetwork()
nn.set_input_layer(X_train.shape[0])
nn.add_hidden_layer(hidden_length=32, activation_function='relu')
nn.add_hidden_layer(hidden_length=16, activation_function='relu')
nn.set_output_layer(1, activation_function='sigmoid')

# Fit the model
nn.fit(X_train, y_train)

# Make predictions
y_pred = nn.predict(X_test)
```

#### Optimizer Options
The `NeuralNetwork` model supports two optimizers: Adam and SGD. You can specify the optimizer during the model initialization:

* To use the Adam optimizer, set `optimizer='adam'`.

* To use the SGD optimizer, set `optimizer='sgd'`.

```python
# Create a model instance using the Adam optimizer
nn_adam = NeuralNetwork(optimizer='adam')

# Create a model instance using the SGD optimizer
nn_sgd = NeuralNetwork(optimizer='sgd')
```

### Decision Tree

```python
from MLmodels.models import DecisionTree

# Create a model instance
dt = DecisionTree()

# Fit the model
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)
```

#### Solver Options

The `DecisionTree` model supports three algorithms:

* To use the ID3 algorithm, set `solver='ID3'`.

* To use the C4.5 algorithm, set `solver='C4.5'`.

* To use the CART algorithm, set `solver='CART'`.

Additionally, you can specify the maximum depth of the tree using the `max_depth` parameter.

```python
# Create a model instance using the ID3 algorithm
dt_id3 = DecisionTree(solver='ID3', max_depth=5)

# Create a model instance using the C4.5 algorithm
dt_c45 = DecisionTree(solver='C4.5', max_depth=5)

# Create a model instance using the CART algorithm
dt_cart = DecisionTree(solver='CART', max_depth=5)
```

### K-means

```python
from MLmodels.models import KMeans

# Create a model instance
km = KMeans(k=5)

# Fit the model
km.fit(X)

# Make predictions
X_classes = km.predict(X)

# Check centroids
centroids = km.centroids
```

#### Initialization Methods

The `KMeans` model supports two initialization methods:

* To use the k-means++ initialization, set `init_method='kmeans++'`.

* To use random initialization, set `init_method='random'`.

## GPU Acceleration

The models now support GPU acceleration, although this feature is currently experimental. It may not perform optimally on certain models. We recommend using GPU acceleration only when dealing with large datasets. 

To disable GPU acceleration, please follow these steps:
1. Open the Python file models.py.
2. Locate the import statement: `import cupy as cp`.
3. Delete or comment out this line.
4. Add the following line after the import statements: `cp = np`.

## Usage Notes

Make sure to install the required dependencies before running the code. You can install them using the following command:

```bash
pip install -r requirements.txt
```
