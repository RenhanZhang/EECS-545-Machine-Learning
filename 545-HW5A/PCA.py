import numpy as np



def PCA(data, d):
    '''
    data is a N * p array, N is number of data points and p the dimention of each data point
    d is the degree of PCA, i.e. number of principle components to be returned
    '''
    (N,p) = data.shape
    mu = np.mean(data, axis=0)
    cov = 1.0/N * np.dot(np.subtract(data, mu).T, np.subtract(data, mu)) #covariance matrix of the data
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    #x,y,z = np.linalg.svd(cov)
    index = eigenvalues.argsort()[::-1][:d]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    return mu, cov, eigenvalues.astype(np.float64), eigenvectors.astype(np.float64)

def recover(target, mu, eigenvecs):
    '''
    produce a vector that's similar to target based on mean vector mu and eigen vectors eigenvecs
    '''
    residual = target - mu;
    recovered = np.copy(mu)
    for i in range(0,eigenvecs.shape[1]):
        recovered = recovered + eigenvecs[:,i] * np.dot(residual, eigenvecs[:,i])
    return recovered

