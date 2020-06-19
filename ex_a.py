import numpy as np
import matplotlib.pyplot as plt
from os import path
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

data = np.genfromtxt(path.join('data/bm.csv'), delimiter=',', dtype=float)

clf = SVC(kernel='rbf', gamma=.5, C=20)  # rbf is a gaussian kernel
y = np.array([data[:, 2]]).T
X = np.array([data[:, 0], data[:, 1]]).T

np.random.seed(7)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]  # shuffle X and y
X_s, y_s = X[:5000, :], y[:5000]  # 5000 samples of all rows, 5000 values of y

clf.fit(X, y.ravel())
score = clf.score(X_s, y_s)  # calc score of comparison
indices = clf.support_  # get indices of support vectors

# Filter
failed = list(filter(lambda point: point[2] == 0, data))
ok = list(filter(lambda point: point[2] == 1, data))
x_f, y_f, z_f = list(zip(*failed))
x_ok, y_ok, z_ok = list(zip(*ok))
plt.xlabel('x')
plt.ylabel('y')

# Mesh
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Mesh Grid
x1, x2 = xx.ravel(), yy.ravel()
pred = np.c_[x1, x2]  # concat x1 and x2
res = clf.predict(pred)
clz_mesh = res.reshape(xx.shape)  # return to mesh format
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # mesh plot
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # colors

# First plot, decision boundary and support vectors
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(X[indices, 0], X[indices, 1], color='r', s=0.5)
plt.show()

# Second plot, decision, support and data points
plt.contour(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(x_ok, y_ok, color='b', s=0.5)
plt.scatter(x_f, y_f, color='r', s=0.5)
plt.show()
