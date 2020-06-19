import numpy as np
import matplotlib.pyplot as plt
from os import path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing as pp

data_train = pd.read_csv(path.join('data/fashion-mnist_train.csv'))
data_test = pd.read_csv(path.join('data/fashion-mnist_test.csv'))

X_train, y_train = np.array(data_train.iloc[:, 1:]), np.array(data_train.iloc[:, 0])
X_test, y_test = np.array(data_test.iloc[:, 1:]), np.array(data_test.iloc[:, 0])

labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Normalize data
scaler = pp.StandardScaler().fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_test = scaler.transform(X_test)


plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(nrows=4, ncols=4)

random = np.random.randint(60000, size=16)  # Get random samples for the training set

for idx in range(1, 17):
    ax = plt.subplot(4, 4, idx)
    ax.imshow(X_train[random[idx - 1], :].reshape(28, 28))  # Reshaping row to 28x28 image
    labelNr = y_train[random[idx - 1]]
    plt.xlabel(labels[labelNr])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

plt.show()

parameters = {
    'hidden_layer_sizes': [[10], [10, 10], [100]],
    'activation': ['relu', 'tanh'],
    'early_stopping': [True, False],
    'alpha': [0.00001, 0.0001, 0.001, 0.1],
    'validation_fraction': [0.1]

}
# grid_search = RandomizedSearchCV(MLPClassifier(verbose=True), cv=3, param_distributions=parameters, n_jobs=3)
# grid_search.fit(Xn_train, y_train)
# clf = grid_search.best_estimator_
# print(grid_search.best_params_)

clf = MLPClassifier(hidden_layer_sizes=[100], early_stopping=True, alpha=0.00001, activation='tanh', random_state=10)
clf.fit(Xn_train, y_train)

pred_test = clf.predict(Xn_test)
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test))
