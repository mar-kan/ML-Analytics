""" regkernel.py """
# %% 
import matplotlib.pyplot as plt
import numpy as np 

def k(x1, x2):
    """
        Calculates a specific 3rd degree polynomial kernel between two points.
        Computes the inner product in a feature space containing 
        linear, quadratic, and cubic cross-terms.
        Args:
            x1 (ndarray): The first input array/value
            x2 (ndarray): The second input array/value
            
        Returns:
            float/int: Scalar representing the kernel similarity
    """

    return np.ndarray.item(x1*x2 + x1*x1*x2*x2 + (x1**3)*(x2**3))

def q1(x):
    """
        Evaluates a constant parametric basis function.
        This function acts as the bias term for the explicit
        parametric part of a semi-parametric regression model. 
        
        Args:
            x: The input data point
            
        Returns:
            int: Always returns exactly 1
    """
    return 1

def q2(x):
    # y = x
    return np.ndarray.item(x)

def kernel_train(k, m, q, ngamma, n, x, y):
    """
        Solves a Semi-Parametric Kernel Regression system.
        Finds the non-linear kernel weights (alpha) and the
        parametric trend weights (d) simultaneously.

        Args:
            k: The kernel function to be used for the non-parametric part
            m: The number of parametric functions
            q: A list of parametric basis functions
            ngamma: The regularization parameter for the kernel part
            n: The number of training samples
            x: The input training data points
            y: The target values for the training data

        Returns:
            ndarray: A combined array of kernel weights (alpha) and parametric weights (d)
    """

    # --- builds Gram Matrix ---
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = k(x[i], x[j])

    # --- builds Q matrix ---        
    Q = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            Q[i,j] = q[j](x[i])

    # --- builds and solves system ---
    M1 = np.hstack((K @ K.T + (ngamma * K), K @ Q)) # regularization term added to the kernel part of the system
    M2 = np.hstack((Q.T @ K.T, Q.T @ Q)) # parametric part of the system
    M = np.vstack((M1,M2)) # final system matrix
    c = np.vstack((K, Q.T)) @ y # constructs target vector for the system
    ad = np.linalg.solve(M,c) # solves the system to find the kernel and parametric weights
    return ad # returns the weights for future predictions
