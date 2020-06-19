from threading import Thread
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.externals import joblib
from sklearn.metrics import multilabel_confusion_matrix

# Fetch MNIST
mnist = fetch_openml('mnist_784', version=1, cache=True)
X, y = mnist['data'], mnist['target']

y = y.astype('float64')  # all y values are chars from the source for some reason..
# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=10)

# Normalize data
scaler = pp.StandardScaler().fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_test = scaler.transform(X_test)

# Instantiate SVC
# rbf = SVC(kernel='rbf', gamma=.001, C=2, probability=True)
# rbf.fit(Xn_train, y_train)
rbf = joblib.load('rbf.model')
y_rbf_pred = rbf.predict(Xn_test)
#print(rbf.score(Xn_test, y_test))
#
# # Find good values for C and gamma.
# C = np.arange(1, 11, 1)
# gamma = np.arange(0.001, 0.01, 0.001)
# param_grid = {'C': C, 'gamma': gamma}
# grid_search = GridSearchCV(rbf, param_grid, scoring='accuracy', n_jobs=10)
# grid_search.fit(Xn_train[:1000, :], y_train[:1000])
#
# print(grid_search.score(Xn_test, y_test))
# print(grid_search.best_params_)


# One vs all
y_0 = np.copy(y_train)
y_1 = np.copy(y_train)
y_2 = np.copy(y_train)
y_3 = np.copy(y_train)
y_4 = np.copy(y_train)
y_5 = np.copy(y_train)
y_6 = np.copy(y_train)
y_7 = np.copy(y_train)
y_8 = np.copy(y_train)
y_9 = np.copy(y_train)

# Make classification binary
y_0[y_train != 0] = 1  # special case, inverse column of prediction for correct comparisons
y_1[y_train != 1] = 0
y_2[y_train != 2] = 0
y_3[y_train != 3] = 0
y_4[y_train != 4] = 0
y_5[y_train != 5] = 0
y_6[y_train != 6] = 0
y_7[y_train != 7] = 0
y_8[y_train != 8] = 0
y_9[y_train != 9] = 0

# Load SVCs
zero_ = joblib.load('models/0.model')
one_ = joblib.load('models/1.model')
two_ = joblib.load('models/2.model')
three_ = joblib.load('models/3.model')
four_ = joblib.load('models/4.model')
five_ = joblib.load('models/5.model')
six_ = joblib.load('models/6.model')
seven_ = joblib.load('models/7.model')
eight_ = joblib.load('models/8.model')
nine_ = joblib.load('models/9.model')

y_pred = np.array([])
# Predict all rows

for row in Xn_test:
    probs = np.array([zero_.predict_proba(row.reshape(1, -1))[0, 0]])
    probs = np.append(probs, [one_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [two_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [three_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [four_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [five_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [six_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [seven_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [eight_.predict_proba(row.reshape(1, -1))[0, 1]])
    probs = np.append(probs, [nine_.predict_proba(row.reshape(1, -1))[0, 1]])
    index = probs.argmax()
    y_pred = np.append(y_pred, index)

# Compare with one-vs-one

# 96.89% accuracy
errors = np.sum(y_pred != y_test)
print()

c1 = multilabel_confusion_matrix(y_test, y_pred)
c2 = multilabel_confusion_matrix(y_test, y_rbf_pred)

print(c1)
print(c2)

