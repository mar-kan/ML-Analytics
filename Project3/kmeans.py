import numpy as np
import matplotlib.pyplot as plt

def kmeans(eps, K, Xmat, c_init):
    n, D = Xmat.shape
    c = c_init
    c_old = np.zeros(c.shape)
    dist2 = np.zeros((K,n))
    while np.abs(c - c_old).sum() > eps:
        c_old = c.copy()
        for i in range(0,K): #compute the squared distances
            dist2[i,:] = np.sum((Xmat - c[:,i].T)**2, 1)        
        label = np.argmin(dist2,0) #assign the points to nearest centroid
        minvals = np.amin(dist2,0)
        for i in range(0,K): # recompute the centroids
            entries = np.where(label == i)
            c[:,i] = np.mean(Xmat[entries,:], 1).reshape(1,2)
    return c, label


Xmat = np.genfromtxt('clusterdata.csv', delimiter=',')
# c_init = np.array([[-2.0, 0.0], [-3.0, -1.0]])

# c_init = np.array([[-4.0, -4.1, -4.2], [0.0, 0.1, 0.2]])
c_init = np.array([[-4.0, -2.0, 10.0], [0.0, -3.0, 10.0]])

# c_init  = np.array([[-2.0,-4,0],[-3,1,-1]])
# c_init = np.array([[-4.0, -2.0, 0.5, 2.0], [0.0, -3.0, -1.5, -1.0]])

eps = 0.001
K = 3
c, label = kmeans(eps, K, Xmat, c_init)

plt.figure(figsize=(10, 6))
plt.scatter(Xmat[:, 0], Xmat[:, 1], c=label, cmap='viridis', alpha=0.6, label='Data Points')
plt.scatter(c[0, :], c[1, :], c='red', marker='X', s=200, label='Cluster Centers')

plt.title('K-means Clustering Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()