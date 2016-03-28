import scipy.io
import numpy as np
from apply_pca import apply_pca
import matplotlib.pyplot as plt


def generate_scatter_plot(X, true_labels):
    # Generate scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    label_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    scas = []
    for (i, cla) in enumerate(label_names):
        class_indices = np.where(true_labels == i + 1)
        class_labels = true_labels[class_indices]
        class_feats = X[class_indices]
        scat = ax.scatter(class_feats[:, 0], class_feats[:, 1], s=30, edgecolor='black', color=plt.cm.Set2(class_labels/10.), vmin=0, vmax=1)
        scas = scas + [scat]

    ax.set_title('PCA scatter plot')
    ax.legend(scas, label_names, loc=4)
    plt.show()
    plt.close(fig)


def main():
    data = scipy.io.loadmat('svhn.mat')
    X_PCA = apply_pca(data['train_features'])
    true_labels = data['train_classes'][0]

    generate_scatter_plot(X_PCA, true_labels)

if __name__ == "__main__":
    main()