import scipy.io
import numpy as np


def my_mean(X):
    my_mean = []
    (m, n) = np.shape(X)

    X_trans = X.T

    for row in X_trans:
        my_mean.append(sum(row)/m)

    # my_mean = np.mean(X, axis=0)
    return np.array(my_mean)  # subtracts mean from columns of matrices


def my_covariance(X):
    (m, n) = np.shape(X)

    X = X - my_mean(X)

    return np.dot(X.T, X) / float(m)


def gaussianMV(mu, det, inv, x):
    x = x - mu

    return -0.5 * np.dot(np.dot(x.T, inv), x) - 0.5 * det  # using log probabilities


def gaussian_full(train_features, train_classes, test_features, test_classes):
    covars = []
    mus = []
    for i in range(1, 11):
        features_classes = train_features[np.where(train_classes == i)]
        covars.append(my_covariance(features_classes))
        mus.append(my_mean(features_classes))

    # Print determinant of each class:
    print "Determinants of covariance matrix of each class:"
    for (i, covar) in enumerate(covars):
        print "Class label {} -> det: {}, log(det): {}".format(i+1, np.linalg.det(covar), np.log(np.linalg.det(covar)))

    # Precompute determinants and inverse of covariance matrix of each class
    dets = [np.log(np.linalg.det(covar)) for covar in covars]
    invs = [np.linalg.inv(covar) for covar in covars]

    # Initialise confusion matrix:
    confusion_matrix = np.zeros((10, 10))

    for i in range(len(test_features)):
        # Compute probabilities for each class
        ps = [gaussianMV(mus[c-1], dets[c-1], invs[c-1], test_features[i]) for c in range(1, 11)]
        predicted_class = np.argmax(ps) + 1
        actual_class = test_classes[i]
        confusion_matrix[actual_class-1][predicted_class-1] += 1

    print "\nConfusion matrix:"
    print confusion_matrix
    print "\nAccuracy: {}".format(np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))


# -- Used to get predictions for grid points when plotting decision boundaries --#
def get_gaussian_full_predictions(train_features, train_classes, test_features):
    covars = []
    mus = []
    for i in range(1, 11):
        features_classes = train_features[np.where(train_classes == i)]
        covars.append(my_covariance(features_classes))
        mus.append(my_mean(features_classes))

    # Precompute determinants and inverse of covariance matrix of each class
    dets = [np.log(np.linalg.det(covar)) for covar in covars]
    invs = [np.linalg.inv(covar) for covar in covars]

    predictions = []

    for i in range(len(test_features)):
        ps = [gaussianMV(mus[c-1], dets[c-1], invs[c-1], test_features[i]) for c in range(1, 11)]
        predicted_class = np.argmax(ps) + 1
        predictions.append(predicted_class)

    return np.array(predictions)


def main():
    data = scipy.io.loadmat('svhn.mat')
    gaussian_full(data['train_features'], data['train_classes'][0], data['test_features'], data['test_classes'][0])

if __name__ == "__main__":
    main()
