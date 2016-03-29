import scipy.io
import numpy as np
import compute_pca
import matplotlib.pyplot as plt

data = scipy.io.loadmat('svhn.mat')

X = data['train_features']

eigenvectors, eigenvalues = compute_pca.compute_pca(X)

cs = np.cumsum(eigenvalues)

x = range(1, 101)
y = cs

print cs

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Cumulative variance')
ax.scatter(x, y)
plt.show()