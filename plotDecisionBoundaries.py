import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plotDecisionBoundaries(train_features, train_classes, X, true_labels, label_names, classifier):
   # Inputs:
   # X : your data to be visualized, N x 2, where N is the number of datapoints you are plotting.
   # true_labels : N sized vector of the true classes of the N test data points 
   # label_names : List of strings with the name of the classes i.e. ['1', '2', '3', ..., '0']
   # classifier : This is the function that you've programmed to classify. It can be k-NN, or any of the two Gaussian classifiers. Note that you can add arguments to plotDecisionBoundaries if you classifier function needs more arguments, or adapt this code as convenient.

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #We first get the min and max values of your features to know the range of the area we are going to color:
    x1 = np.min(X[:,0])
    x2 = np.max(X[:,0])

    y1 = np.min(X[:,1])
    y2 = np.max(X[:,1])

    # Stepsize defines how fine-grained we want our grid. The small the
    # value, the more resolution the visualization will have, at the
    # expense of computational cost (we would need to classify more
    # data-points since the grid would be denser).
    stepsize = 0.1

    #We obtain the grid vectors for the two dimensions we are plotting. 
    xx, yy = np.meshgrid(np.arange(x1, x2, stepsize), np.arange(y1, y2, stepsize))

    #We stack both dimensions into a matrix with two columns that will be fed to the classifier.
    gridX = np.c_[xx.ravel(), yy.ravel()]

    # Classify every point in the grid. Adapt this part to your code of the classifiers:
    grid_labels = classifier(train_features, train_classes, gridX)

    # Reshape into the rectangle shape to be used in pcolormesh:
    grid_labels_reshaped = grid_labels.reshape(xx.shape)

    # Here we colour the regions with with the color corresponding to the class. We use the Set2 colormap but other color maps can be chosen.
    ax.pcolormesh(xx, yy, grid_labels_reshaped, cmap=matplotlib.cm.Set2, vmin=0, vmax=10)

    #Show the per-class scatter plots and use the proper color map.
    scas = []
    for (i,cla) in enumerate(label_names):
        classIndices = np.where(true_labels == i + 1)
        classLabels = true_labels[classIndices]
        classFeats = X[classIndices]
        scat = ax.scatter(classFeats[:, 0], classFeats[:, 1], s=30, edgecolor='black', color=matplotlib.cm.Set2(classLabels/10.), vmin=0, vmax=1)
        scas = scas + [scat]

    ax.legend(scas,label_names,loc=4)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision boundary')

    #Uncomment this line if you want to save the figure in a path of your choosing. 
    # fig.savefig('path_to_your_figure.png', bbox_inches='tight')
    plt.show()

    plt.close(fig)