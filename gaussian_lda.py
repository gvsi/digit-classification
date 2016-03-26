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


def lda(mu, covar, x):
    (d, b) = np.shape(covar)

    mu = np.reshape(mu, d, 1)
    x = np.reshape(x, d, 1)

    wkT = np.dot(mu.T, np.linalg.inv(covar))
    return np.dot(wkT, x) - 0.5 * np.dot(wkT, mu)  # ignores likelihood


covars = []
mus = []
for i in range(1, 11):
    features_classes = data['train_features'][np.where(data['train_classes'][0] == i)]
    covars.append(my_covariance(features_classes))
    mus.append(my_mean(features_classes))

shared_covar = np.sum(covars, 0) / 10


confusion_matrix = np.zeros((10, 10))


for i in range(len(data['test_features'])):
    actual_class = data['test_classes'][0][i]
    ps = [lda(mus[c-1], shared_covar, data['test_features'][i]) for c in range(1, 11)]
    predicted_class = np.argmax(ps) + 1
    confusion_matrix[actual_class-1][predicted_class-1] += 1
    # print gaussianMV(mm, mc, data['test_features'][i])

print confusion_matrix
print("Accuracy: ", np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))

# ps = np.array(ps)
# print ps[np.where(data['test_classes'][0] == 1)]
