import scipy.io
import numpy as np
import compute_pca
import matplotlib.pyplot as plt

data = scipy.io.loadmat('svhn.mat')

X = data['train_features']

eigen_vectors, eigen_values = compute_pca.compute_pca(X)

print(eigen_values)

cs = np.cumsum(eigen_values)

print cs

x = range(1, 101)
y = cs

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y)
plt.show()