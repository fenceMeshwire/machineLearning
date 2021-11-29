# Dependencies
from sklearn.linear_model import LinearRegression
import numpy as np

# Some linear data, creating a numpy array:
data = np.array([200, 210, 220])
n = len(data)

# process the data with linear regression:
model = LinearRegression().fit(np.arange(n).reshape((n, 1)), data)

# print the predicted data from linear regression:
print(model.predict([[3], [4]]))
