import scipy.io
import numpy as np
data = scipy.io.loadmat('svhn.mat')


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


def gaussianMV(mu, covar, x):
    (d, b) = np.shape(covar)

    mu = np.reshape(mu, d, 1)
    x = np.reshape(x, d, 1)

    x = x - mu

    return 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(covar)) * np.exp(-0.5 * np.dot(np.dot(x.T, np.linalg.inv(covar)), x))


def gaussian_full(train_features, train_classes, test_features, test_classes):
    covars = []
    mus = []
    for i in range(1, 11):
        features_classes = train_features[np.where(train_classes == i)]
        covars.append(my_covariance(features_classes))
        mus.append(my_mean(features_classes))

    confusion_matrix = np.zeros((10, 10))

    for i in range(len(test_features)):
        actual_class = test_classes[i]
        ps = [gaussianMV(mus[c-1], covars[c-1], test_features[i]) for c in range(1, 11)]
        predicted_class = np.argmax(ps) + 1
        confusion_matrix[actual_class-1][predicted_class-1] += 1

    print confusion_matrix
    print("Accuracy", np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))


# -- Used to get predictions for grid points when plotting decision boundaries --#
def get_gaussian_full_predictions(train_features, train_classes, test_features):
    covars = []
    mus = []
    for i in range(1, 11):
        features_classes = train_features[np.where(train_classes == i)]
        covars.append(my_covariance(features_classes))
        mus.append(my_mean(features_classes))

    predictions = []

    for i in range(len(test_features)):
        ps = [gaussianMV(mus[c-1], covars[c-1], test_features[i]) for c in range(1, 11)]
        predicted_class = np.argmax(ps) + 1
        predictions.append(predicted_class)

    return np.array(predictions)


def main():
    gaussian_full(data['train_features'], data['train_classes'][0], data['test_features'], data['test_classes'][0])

if __name__ == "__main__":
    main()
