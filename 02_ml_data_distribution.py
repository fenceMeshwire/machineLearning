#!/usr/bin/env python3

# Python 3.9.5

# 02_ml_data_distribution.py

# Dependencies
import numpy as np
from sklearn.datasets import make_classification

# Randomize the order of the full dataset.
x, y = make_classification(n_samples=20000, weights=(0.9, 0.1))
index = np.argsort(np.random.random(y.shape[0]))

X = x[index]
Y = y[index]

# TRAINING DATA is 90 percent of the total dataset:   TRAIN
# TEST DATA is 5 percent of the total dataset:        TEST
# VALIDATION DATA is 5 percent of the total dataset:  VALIDATE

n_train = int(0.9 * X.shape[0])
n_val = int(0.05 * Y.shape[0])

# Associate samples with the training set.
x_train = X[:n_train]
y_train = Y[:n_train]

# Associate samples with the validation set.
x_val = X[n_train:(n_train + n_val)]
y_val = Y[n_train:(n_train + n_val)]

# Associate samples with the test set.
x_test = X[(n_train + n_val):]
y_test = Y[(n_train + n_val):]
