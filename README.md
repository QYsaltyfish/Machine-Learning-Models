# Machine Learning Models from Scratch in Python

## Overview

This repository contains implementations of various machine learning models from scratch in Python.

## Models Included

1. **Multiple Linear Regression**: Including Lasso & Ridge regularization techniques.
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**: Implementations of both hard-margin and soft-margin SVM.
4. **Neural Network**
5. **Decision Tree**: ID3 & C4.5 algorithms (post-pruning to be implemented).

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
