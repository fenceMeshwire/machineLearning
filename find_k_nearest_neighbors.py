#!/usr/bin/env python3

# find_k_nearest_neighbors.py

# Purpose:
# a) Find the k nearest neighbors of a predefined numpy array
# b) Unify the k nearest neighbors into a single prediction

# Dependencies:
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

# Data (Living space (square meters) / Price (thousand Euro))
values_results = np.array([[35, 200], [45, 250], [40, 225],
[35, 190], [25, 165], [40, 215],
[32, 200], [45, 265], [38, 200]])

# One-liner operation for x neighbors:
x = len(values_results) # x for all neighbors
# x = 3 # x for three nearest neighbors
knn = KNeighborsRegressor(n_neighbors=x).fit(values_results[:,0].reshape(-1,1), values_results[:,1])

# Predict the result
value = 28
result = knn.predict([[value]])
print('Value:', value, 'Result:', result)

# Data visualization with matplotlib.pyplot
for value_result in values_results:
    plt.scatter(value_result[0], value_result[1], c='blue')

# Plot the predicted result
plt.xlabel('Size in square meters [m^2]', loc='center') # Label for x axis
plt.ylabel('Price in currency [1k €]', loc='center') # Label for y axis

# Plot the predicted result with a label (legend).
plt.plot(value, result, marker='o', c='red', ls='', label="predicted result")
plt.legend(loc='upper left')
plt.show()
