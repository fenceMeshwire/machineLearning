#!/usr/bin/env python3

# applyKMeans.py

# Purpose: 
# Clustering data with the k-Means method
# Visualization of the data with matplotlib.pyplot

# Dependencies:
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Data (input / output)
input_output = np.array([[200, 90000], [100, 40000], [80, 32000], 
[190, 87000], [185, 91000], [90, 34000],
[210, 92000], [95, 42000], [82, 40000]])

# The model will be trained with 2 cluster centers
k_means = KMeans(n_clusters=2).fit(input_output)

results = k_means.cluster_centers_
print(results)

# Data visualization with matplotlib.pyplot, red color for cluster centers
for data in input_output:
    plt.scatter(data[0], data[1], c='blue')
for result in results:
    plt.scatter(result[0], result[1], c='red')

plt.show()
