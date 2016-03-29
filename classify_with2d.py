import scipy.io

from knn import knn
from gaussian_full import gaussian_full
from gaussian_lda import gaussian_lda
from apply_pca import apply_pca

data = scipy.io.loadmat('svhn.mat')
train_features = data['train_features'][:, :2]  # use only first two features
test_features = data['test_features'][:, :2]  # use only first two features
train_classes = data['train_classes'][0]
test_classes = data['test_classes'][0]

# Uncomment the following two lines to use PCA
# train_features = apply_pca(data['train_features'])
# test_features = apply_pca(data['test_features'])

# ---- Simulation using online two features ---- #
print "Classifying with kNN..."
knn(train_features, train_classes, test_features, test_classes, 1)

print "\nClassifying with Gaussian models (full covariance matrix per class)..."
gaussian_full(train_features, train_classes, test_features, test_classes)

print "\nClassifying with Gaussian models and LDAs (share covariance matrix)..."
gaussian_lda(train_features, train_classes, test_features, test_classes)


# ---- Simulation using full two feature space ---- #
train_features = data['train_features']
test_features = data['test_features']

print "\nClassifying with kNN..."
knn(train_features, train_classes, test_features, test_classes, 1)

print "\nClassifying with Gaussian models (full covariance matrix per class)..."
gaussian_full(train_features, train_classes, test_features, test_classes)

print "\nClassifying with Gaussian models and LDAs (share covariance matrix)..."
gaussian_lda(train_features, train_classes, test_features, test_classes)
