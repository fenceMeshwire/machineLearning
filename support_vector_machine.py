#!/usr/bin/env python3

# Python 3.9.5

# support_vector_machine.py

# Dependencies
from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets._samples_generator import make_blobs
from sklearn.svm import SVC 

def plot_svc_decision_function(model, ax=None, plot_support=True):
    
    if ax is None:
        ax = plt.gca()      # get current polar axes
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    x = np.linspace(x_lim[0], x_lim[1], 10)             # x, y, z with z > 1
    y = np.linspace(y_lim[0], y_lim[1], 10)             # x, y, z with z > 1
    Y, X = np.meshgrid(y, x)                            # Return coordinate matrices from coordinate vectors.
    x_y = np.vstack([X.ravel(), Y.ravel()]).T           # Transpose meshgrid, stack arrays in sequence vertically (row wise).
    P = model.decision_function(x_y).reshape(X.shape)   # Reshape model array.

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], 
                alpha=0.5, linestyles=['--', '-', '--']);

    # plot circles around support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], 
                    model.support_vectors_[:, 1], 
                    s=200, linewidths=1, 
                    facecolors='None', edgecolors='black')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def make_svc_model():
    # Parameters:
    n_samples = 100         # Number of samples
    centers = 2             # Number of groups/centers
    random_state = 0        # Randomized data
    cluster_std = 0.45      # 0 < cluster_std < 1
    colormapping = 'winter' # Defines the colors of the scatter plotted data groups
    edgecolors = 'black'    # Black circles around the scatter plotted data groups
    # Create data with parameters
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
        random_state=random_state, cluster_std=cluster_std);
    # Draw scatterplot with data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, 
                cmap=colormapping, edgecolors=edgecolors);
    # Make Support Vector Machines model
    svc_model = SVC(kernel='linear', C=1E10)
    return svc_model.fit(X, y)


if __name__ == '__main__':

    svc_model = make_svc_model()            # Get Support Vector Machines model.
    plot_svc_decision_function(svc_model)   # Plot model.
    plt.show()                              # Show model and data.
