import scipy.io
import numpy as np
data = scipy.io.loadmat('svhn.mat')


def mean_normalise(X):
    my_mean = []
    (m, n) = np.shape(X)

    X_trans = np.transpose(X)

    for row in X_trans:
        my_mean.append(sum(row)/m)

    # my_mean = np.mean(X, axis=0)

    # print X[0,0]
    # print np.shape(my_mean)
    # print my_mean[0]
    # # print np.shape(my_mean)
    # # print my_mean[:, np.newaxis];
    # print (X - my_mean)[0,0]
    return X - my_mean  # subtracts mean from columns of matrices


def compute_pca(X):
    (m, n) = np.shape(X)

    normalised_data = mean_normalise(X)

    # print(np.shape(X))
    # print np.transpose(normalised_data)

    # print np.shape(X), np.shape(np.transpose(X)), np.shape(np.dot(np.transpose(X), X))

    # Compute eig of covariance matrix (vectorised)
    # evals, evecs = np.linalg.eig(np.cov(normalised_data, rowvar=0))
    evals, evecs = np.linalg.eig(np.dot(normalised_data.T, normalised_data) * (1/float(m-1)))

    # evecs = evecs.T

    # Sort eigenvectors
    idx = evals.argsort()[::-1]
    eigenvalues = evals[idx]
    eigenvectors = evecs[:, idx]

    eigenvectors = eigenvectors.T

    for i in range(len(eigenvectors)):
        if eigenvectors[i][0] < 0:
            eigenvectors[i] *= -1


    # print np.shape(v)

    # print(np.mean(X, axis=0))
    return eigenvectors, eigenvalues

# print mean_normalise(data['train_features'])
# compute_pca(data['train_features'])
