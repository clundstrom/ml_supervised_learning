import numpy as np
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(2,))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

clf.fit(X, y)

res = clf.predict(np.array([[0, 1]]))
weights = clf.coefs_
bias = clf.intercepts_
print()
