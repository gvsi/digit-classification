import scipy.io
import numpy as np


def my_mean(X):
    my_mean = []
    (m, n) = np.shape(X)

    X_trans = X.T

    for row in X_trans:
        my_mean.append(sum(row)/m)

    return np.array(my_mean)  # subtracts mean from columns of matrices


def my_covariance(X):
    (m, n) = np.shape(X)

    X = X - my_mean(X)

    return np.dot(X.T, X) / float(m)


def lda(mu, shared_cov_inv, x):
    wkT = np.dot(mu.T, shared_cov_inv)
    return np.dot(wkT, x) - 0.5 * np.dot(wkT, mu)  # ignores likelihood


def gaussian_lda(train_features, train_classes, test_features, test_classes):
    covars = []
    mus = []
    for i in range(1, 11):
        features_classes = train_features[np.where(train_classes == i)]
        covars.append(my_covariance(features_classes))
        mus.append(my_mean(features_classes))

    shared_covar = np.sum(covars, 0) / 10
    shared_cov_inv = np.linalg.inv(shared_covar)

    # Print determinant of shared covariance matrix:
    print "Determinant of shared covariance matrix: \ndet: {}, log(det): {}".format(np.linalg.det(shared_covar), np.log(np.linalg.det(shared_covar)))

    # Initialise confusion matrix:
    confusion_matrix = np.zeros((10, 10))

    for i in range(len(test_features)):
        actual_class = test_classes[i]
        ps = [lda(mus[c-1], shared_cov_inv, test_features[i]) for c in range(1, 11)]
        predicted_class = np.argmax(ps) + 1
        confusion_matrix[actual_class-1][predicted_class-1] += 1

    print "\nConfusion matrix:"
    print confusion_matrix
    print "\nAccuracy: {}".format(np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))


# -- Used to get predictions for grid points when plotting decision boundaries --#
def get_gaussian_lda_predictions(train_features, train_classes, test_features):
    covars = []
    mus = []
    for i in range(1, 11):
        features_classes = train_features[np.where(train_classes == i)]
        covars.append(my_covariance(features_classes))
        mus.append(my_mean(features_classes))

    shared_covar = np.sum(covars, 0) / 10
    shared_cov_inv = np.linalg.inv(shared_covar)

    predictions = []

    for i in range(len(test_features)):
        ps = [lda(mus[c-1], shared_cov_inv, test_features[i]) for c in range(1, 11)]
        predicted_class = np.argmax(ps) + 1
        predictions.append(predicted_class)

    return np.array(predictions)


def main():
    data = scipy.io.loadmat('svhn.mat')
    gaussian_lda(data['train_features'], data['train_classes'][0], data['test_features'], data['test_classes'][0])

if __name__ == "__main__":
    main()
