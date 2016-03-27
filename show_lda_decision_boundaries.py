import scipy.io
from plotDecisionBoundaries import plotDecisionBoundaries
from gaussian_lda import get_gaussian_lda_predictions

data = scipy.io.loadmat('svhn.mat')
train_features = data['train_features'][:, :2]  # use only first two features
test_features = data['test_features'][:, :2]  # use only first two features
train_classes = data['train_classes'][0]
test_classes = data['test_classes'][0]

plotDecisionBoundaries(train_features, train_classes, test_features, test_classes, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], get_gaussian_lda_predictions)