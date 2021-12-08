#!/usr/bin/env python3

# Logistic-Regression.py

# Dependencies:
from sklearn.linear_model import LogisticRegression
import numpy as np

# Data (complaints, frustrated)
X = np.array([[0, "No"],
              [10, "No"],
              [15, "No"],
              [25, "Yes"],
              [35, "Yes"],
              [60, "Yes"]])

n = len(X)

# Building the model for the Logistic Regression:
model = LogisticRegression().fit(X[::,0].reshape(n, 1), X[::,1])

# Making a prediction:
print(model.predict([[2],[7],[17],[36],[55]]))

# Showing the transition from "No" to "Yes":
for i in range(10, 30):
    print("x=" + str(i) + " --> " + str(model.predict_proba([[i]])))
