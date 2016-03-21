import scipy.io
import numpy as np

data = scipy.io.loadmat('svhn.mat')

train_features = data['train_features']
test_features = data['test_features']
train_classes = data['train_classes']
test_classes = data['test_classes']


def euclidean_distances(v1, v2):
    d = ((v1 - v2) ** 2).sum()
    return np.sqrt(d)


def find_modes(lst):
    max_freq = max(map(lst.count, lst))
    modes = [i for i in lst if lst.count(i) == max_freq]
    return list(set(modes))


def knn(data, k):
    # distances = []
    # for x in train_features:
    #     distances.append(euclidean_distances(x, data))
    (train_features - data) ** 2
    distances = np.sqrt(np.sum(((train_features - data) ** 2), axis=1))

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

# knn(test_features[999])

confusion_matrix = np.zeros((10, 10))
actual_classes = test_classes[0]

for i in range(len(test_features)):
    predicted_class = knn(test_features[i], 1)
    confusion_matrix[actual_classes[i]-1][predicted_class-1] += 1

print(confusion_matrix)
