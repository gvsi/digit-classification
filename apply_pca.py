import scipy.io
import numpy as np
from compute_pca import compute_pca


def apply_pca(X):

    eigenvectors, eigenvalues = compute_pca(X)

    # Stack first two eigenvectors
    E_2 = np.column_stack((eigenvectors[0], eigenvectors[1]))

    X_PCA = np.dot(X, E_2)

    # Print report data
    print "Largest two eigenvalues:"
    print eigenvalues[0]
    print eigenvalues[1]

    print "\n"

    print "First five rows of E_2:"
    print E_2[0:5, :]

    print "\n"

    print "First five rows of X_PCA:"
    print X_PCA[0:5, :]

    return X_PCA


def main():
    data = scipy.io.loadmat('svhn.mat')
    apply_pca(data['train_features'])

if __name__ == "__main__":
    main()