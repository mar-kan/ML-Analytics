""" polynomial_regression.py """

import numpy as np
from numpy.linalg import norm , solve
import matplotlib.pyplot as plt
from numpy.random import rand , randn


def generate_data_even(p, beta , sig, n, mu):
   """
   Generates data with an even spread
   Args:
      - p (int): Polynomial degree
      - beta (nparray): Coefficients for each polynomial
      - sig (int): Standard deviation for noise
      - n (int): number of samples to generate

   Returns:
      - u (nparray): Array containing the generated data
      - y (nparray): Array containing the targets
   """

   # creates an array of length n and populates it with random samples from a uniform distribution over [0, 1)
   u = np.random.rand(n, 1) + mu

   # (u ** np.arange(0, p+1)): Creates model matrix where each column is u raised to a power (0 to p)
   # @ beta: Performs matrix multiplication to apply the polynomial coefficients
   # + sig * np.random.randn: Adds Gaussian noise to each sample
   y = (u ** np.arange(0, p+1)) @ beta + sig * np.random.randn(n, 1)
   return u, y

def generate_data_skewed(p, beta , sig, n, mu, sigu):
   """
   Generates skewed data 
   Args:
      - p (int): Polynomial degree
      - beta (nparray): Coefficients for each polynomial
      - sig (int): Standard deviation for noise
      - n (int): number of samples to generate

   Returns:
      - u (nparray): Array containing the generated data
      - y (nparray): Array containing the targets
   """

   # creates an array of length n and populates it with random samples from a normal distribution
   u = mu + sigu * np.random.randn(n, 1)

   # (u ** np.arange(0, p+1)): Creates model matrix where each column is u raised to a power (0 to p)
   # @ beta: Performs matrix multiplication to apply the polynomial coefficients
   # + sig * np.random.randn: Adds Gaussian noise to each sample
   y = (u ** np.arange(0, p+1)) @ beta + sig * np.random.randn(n, 1)
   return u, y

def model_matrix(p, u):
   """
   Constructs model matrix for polynomial regression
   Args:
      - p (int): Polynomial Degree
      - u (nparray): Array of generated data

   Returns:
      X (nparray): Model matrix [n, p+1]
   """
   X = np.ones((len(u), 1)) # creates 1st column of matrix with 1s (u^0) with length n (len(u))

   p_range = np.arange(0, p + 1)    
   for p_current in p_range: # creates the rest of the columns of the matrix with u^p
      if p_current > 0:
         X = np.hstack((X, u**(p_current))) # stacks all columns together in X
   return X
             
def train(X, y):
   """
    Solves the equation: (X^T * X) * beta = X^T * y
    to find the coefficients that minimize the loss (squared error)

    Args:
        - X (nparray): Model matrix
        - y (nparray): Target values

    Returns:
        - betahat (nparray): Estimated coefficients vector [p+1, 1]
    """
   
   betahat = solve(X.T @ X, X.T @ y)
   return betahat

def test_coefficients(n, betahat, X, y):
   """
   Evaluates the model by calculating the Mean Squared Error (MSE) of the model predictions.

    Args:
        - n (int): Number of samples in the dataset
        - betahat (nparray): Estimated coefficients vector
        - X (nparray): Model matrix
        - y (nparray): Target values

    Returns:
        - loss (float): MSE value
    """
   
   y_hat = X @ betahat # generates model's predictions
   loss = (norm(y - y_hat)**2/n) # calculates the average squared euclidean distance between targets and predictions 
   return loss
