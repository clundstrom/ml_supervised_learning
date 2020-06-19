import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


def createMesh(X, classifier, plt, stepsize=0.02):
    """
    Plots a mesh with a default stepsize of .02
    :param X: data ( x,y)
    :param classifier: Classifier used to predict mesh
    :param plt: matplotlib.plt
    """
    h = stepsize # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Mesh Grid
    x1, x2 = xx.ravel(), yy.ravel()
    mesh = np.c_[x1, x2]  # concat x1 and x2
    res = classifier.predict(mesh)
    clz_mesh = res.reshape(xx.shape)  # return to mesh format
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#c242f5'])  # mesh plot
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # colors

    # Plot mesh
    plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
    # Contour decision boundary
    #plt.contour(xx, yy, clz_mesh, cmap=cmap_bold)


def pickParams(svc: SVC, gamma, c):
    """
    Aids gridsearch in choosing parameters for SVM.
    :param svc: SVM
    :param gamma: param
    :param c: param
    :return: tuned SVM
    """
    svc = svc
    if svc.kernel == 'linear':
        svc.C = c
    elif svc.kernel == 'poly':
        svc.C = c
        svc.D = gamma
    elif svc.kernel == 'rbf':
        svc.C = c
        svc.gamma = gamma
    return svc


def gridSearch(X_train, y_train, X_val, y_val, grid, svc: SVC):
    """
    :param X_train: Training set
    :param y_train: output training set
    :param X_val: Validation set
    :param y_val: output validation set
    :param grid: A grid of paramter values to be tested
    :param svc: Vector machines
    :return: Best parameters for respective SVM
    """
    previous_best = 0
    for gamma in grid[:, 0]:
        for c in grid[:, 1]:
            svc = pickParams(svc, gamma, c)
            svc.fit(X_train, y_train.ravel())
            score = svc.score(X_val, y_val)

            # Check best score
            if (score > previous_best):
                best_C = c
                best_gamma = gamma
                previous_best = score

    return best_C, best_gamma