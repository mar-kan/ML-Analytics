""" polynomial_regression_CV.py """
from polynomial_regression import *

def linear_cv(K, n, X, y):
    loss = []
    for k in range(1, K+1):        
        # integer indices of test samples        
        test_ind = ((n/K)*(k-1) + np.arange(1, n/K + 1) - 1).astype('int')
        train_ind = np.setdiff1d(np.arange(n), test_ind)
    
        X_train, y_train = X[train_ind, :], y[train_ind, :]
        X_test, y_test = X[test_ind, :], y[test_ind]
        # fit model and evaluate test loss
        betahat = train(X_train,y_train)
        loss.append(test_coefficients(n, betahat, X_test, y_test))
        cv = sum(loss) / n
    return cv
