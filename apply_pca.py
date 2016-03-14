import scipy.io
import numpy as np
import compute_pca
import matplotlib.pyplot as plt


data = scipy.io.loadmat('svhn.mat')

X = data['train_features']
# print(compute_pca.data['train_features'])
print X

eVecs, eVals = compute_pca.compute_pca(X)

# print eVals[0], eVals[1]
E_2 = np.column_stack((eVecs[0], eVecs[1]))

X_PCA = X.dot(E_2)
# print X.dot(E_2)

x = X_PCA[:, 0]
y = X_PCA[:, 1]

# print(np.shape(np.transpose(data['train_classes'])))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, c=data['train_classes'])
plt.show()