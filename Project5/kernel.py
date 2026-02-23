""" kernel.py """
from genham import hammersley
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import norm


def peaks(x,y):
    """ Generates a surface with hills and valleys representing the ground truth for the training dataset """

    z =  (3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) 
          - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2)
          - 1/3 * np.exp(-(x+1)**2 - y**2)) 
    return(z)

def kernel_train(k, n, x, y):
    """ Training function to calculate weights that map the inputs to the targets """

    # fills Gram Matrix K
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = k(x[i,:],x[j]) # measures similarity between each point with all other points
    
    # solves linear equation system to find the coefficients (alpha) for future predictions
    alpha = np.linalg.solve(K@K.T, K@y)
    return K, alpha

sig2 = 0.3 # kernel parameter

def k(x,u):
    """
        Gaussian Kernel function
        Calculates point similarity based on their Euclidean distance.
    """

    return(np.exp(-0.5*norm(x- u)**2/sig2))

def k_linear(x, u):
    """ Linear kernel function """

    return np.dot(x, u)

def k_poly(x, u, c=1.0, p=15):
    """ Polynomial feature kernel function """

    return (np.dot(x, u) + c)**p
