import scipy.io
import numpy as np


def my_mean(X):
    my_mean = []
    (m, n) = np.shape(X)

    X_trans = np.transpose(X)

    for row in X_trans:
        my_mean.append(sum(row)/m)

    return my_mean


def compute_pca(X):
    (m, n) = np.shape(X)

    X = X - my_mean(X)

    evals, evecs = np.linalg.eig(np.dot(X.T, X) * (1/float(m-1)))

    # Sort eigenvectors
    idx = evals.argsort()[::-1]
    eigenvalues = evals[idx]
    eigenvectors = evecs[:, idx]

    eigenvectors = eigenvectors.T

    # Invert sign of eigenvectors with negative first element
    for i in range(len(eigenvectors)):
        if eigenvectors[i][0] < 0:
            eigenvectors[i] *= -1

    return eigenvectors, eigenvalues


def main():
    data = scipy.io.loadmat('svhn.mat')
    evecs, evals = compute_pca(data['train_features'])
    print evals

if __name__ == "__main__":
    main()