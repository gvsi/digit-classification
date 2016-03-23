import scipy.io
import numpy as np

data = scipy.io.loadmat('svhn.mat')

train_features = data['train_features']
test_features = data['test_features']
train_classes = data['train_classes']
test_classes = data['test_classes']


def find_modes(lst):
    max_freq = max(map(lst.count, lst))
    modes = [i for i in lst if lst.count(i) == max_freq]
    return list(set(modes))


def classify(v, k):
    distances = np.sqrt(np.sum(((train_features - v) ** 2), axis=1))

    distances = np.array(distances)
    idx = distances.argsort()[::1]
    # distances = distances[idx]
    # sorted_X = train_features[idx, :]
    labels_X = np.array(train_classes[0])[idx]

    modes = find_modes(list(labels_X)[:k])

    if len(modes) == 1:
        return modes[0]
    else:
        return labels_X[0]


def knn(k):
    confusion_matrix = np.zeros((10, 10))
    actual_classes = test_classes[0]

    for i in range(len(test_features)):
        predicted_class = classify(test_features[i], k)
        confusion_matrix[actual_classes[i]-1][predicted_class-1] += 1

    print(confusion_matrix)
    print("Accuracy: ", np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))

knn(1)
