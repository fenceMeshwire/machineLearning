#!/usr/bin/env python3

# Python 3.9.5

# 01_check_data_validation.py

# Dependencies
import numpy as np
import matplotlib.pyplot as plt

# Some random data:
data = np.random.randint(-50, 50, (20, 5))
data = np.array(data)

# Analyse the given data:
np.min(data)                    # Minimum
np.max(data)                    # Maximum
np.max(data) - np.min(data)     # Range
np.mean(data)                   # Mean
np.std(data)                    # Standard deviation

def standard_error(mean, std):
    return float(np.std(data)) / float((len(data)) ** 0.5)

standard_error(np.mean, np.std) # Standard error

# Visualize the data:
plt.boxplot(data)               # Provide box plot
plt.show()                      # Show box plot
