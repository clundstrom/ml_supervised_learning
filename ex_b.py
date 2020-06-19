import numpy as np
import matplotlib.pyplot as plt
from os import path
from sklearn.tree import DecisionTreeClassifier
import utils

data = np.genfromtxt(path.join('data/artificial.csv'), delimiter=',', dtype=float)

X, y = np.array(data[:, 0:2]), np.array(data[:, 2])

# Normalize X
#scaler = pp.StandardScaler().fit(X)
#X_train = scaler.transform(X)

# Filter data
failed = data[data[:, 2] == 0]
ok = data[data[:, 2] == 1]

# Train classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

utils.createMesh(X, clf, plt, 1)
plt.xlim()

# First plot, decision boundary and support vectors
plt.scatter(ok[:, 0], ok[:, 1], color='b', s=0.5)
plt.scatter(failed[:, 0], failed[:, 1], color='r', s=0.5)

plt.show()
print()
