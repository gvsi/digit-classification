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

    return np.dot(X.T, X) / float(m-1)

# print my_covariance(data['train_features'])
#
# print np.cov(data['train_features'], rowvar=0)

# print np.where(data['train_classes'][0] == 1)


def gaussianMV(mu, covar, x):
    (d, b) = np.shape(covar)

    mu = np.reshape(mu, d, 1)
    x = np.reshape(x, d, 1)

    x = x - mu

    return 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(covar)) * np.exp(-0.5 * np.dot(np.dot(x.T, np.linalg.inv(covar)),x))


covars = []
mus = []
for i in range(1, 11):
    features_classes = data['train_features'][np.where(data['train_classes'][0] == i)]
    covars.append(my_covariance(features_classes))
    mus.append(my_mean(features_classes))

X1 = data['train_features'][np.where(data['train_classes'][0] == 1)]
print np.shape(X1)
# print(X1)
# print my_covariance(X1)
# print np.cov(X1, rowvar=0)

# print np.reshape(my_mean(X1), 100, 1)
# print np.shape(my_covariance(X1))


mc = my_covariance(X1)
mm = my_mean(X1)


confusion_matrix = np.zeros((10, 10))


for i in range(len(data['test_features'])):
    actual_class = data['test_classes'][0][i]
    ps = [gaussianMV(mus[c-1], covars[c-1], data['test_features'][i]) for c in range(1, 11)]
    predicted_class = np.argmax(ps) + 1
    confusion_matrix[actual_class-1][predicted_class-1] += 1
    # print gaussianMV(mm, mc, data['test_features'][i])

print confusion_matrix
print("Accuracy: ", np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))

# ps = np.array(ps)
# print ps[np.where(data['test_classes'][0] == 1)]
