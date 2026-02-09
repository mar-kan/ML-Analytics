import numpy as np
from scipy.stats import multivariate_normal


def exp_max(Iter, K, pdf, train, Xmat, W_Init, P_Init):
    """
    Performs the Expectation-Maximization (EM) algorithm to estimate parameters.

    Args:
        - Iter (int): number of iterations to run algorithm
        - K (int): number of clusters
        - pdf (function): probability density function (calculates likelihood)
        - train (function): function used to train algorithm (updates parameters)
        - Xmat (ndarray): training data matrix of shape (n, D)
        - W_Init (ndarray): initial cluster weights of shape (1, K)
        - P_Init (ndarray): initial parameters for the PDF of shape (D, K)

    Returns:
        - W (ndarray): final cluster weights of shape (1, K)
        - P (ndarray): final estimated parameters of shape (D, K)
        - p (ndarray): probability (responsibility) matrix of shape (K, n)
    """
    n, D = Xmat.shape
    p = np.zeros((K, n))
    W, P = W_Init, P_Init
    for i in range(0, Iter):
        # E-Step
        for k in range(0, K):
            # calculates weighted likelihood of each data point belonging to cluster k
            p[k,:] = W[0,k] * pdf(P[:,k], Xmat)

        # M-Step
        # normalizes columns so each data point's responsibilities sum to 1
        p = (p/sum(p,0))

        # updates weights by average responsability per cluster
        W = np.mean(p,1).reshape(1,K)

        # updates estimated parameters per cluster
        for k in range(0,K):
            P[:,k] = train(p[k,:],Xmat)
        
    return W, P, p

def normal_train(p, Xmat):
    """
    Calculates the weighted mean for a cluster based on responsibilities.

    Args:
        - p (ndarray): responsibility vector for a single cluster of shape (n, 1)
        - Xmat (ndarray): training data matrix of shape (n, D)

    Returns:
        - m (ndarray): updated mean vector of shape (D, 1)
    """

    m = (Xmat.T @ p.T) / sum(p)
    return m

def normal_pdf(m, Xmat):
    """
    Calculates the Multivariate Normal density for the data

    Args:
        - m (ndarray): mean vector of shape (D, 1)
        - Xmat (ndarray): training data matrix of shape (n, D)

    Returns:
        - (ndarray): vector of density values for each sample of shape (n, 1)
    """

    var = 1

    # supports 2x2 matrices
    C = np.zeros((2, 2))
    C[0, 0] = var
    C[1, 1] = var
    
    # evaluates the normal distribution at each point in Xmat
    mvn = multivariate_normal(m.T, C)
    return mvn.pdf(Xmat)
