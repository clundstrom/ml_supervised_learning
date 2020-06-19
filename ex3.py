import numpy as np
from os import path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing as pp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

data_train = np.genfromtxt(path.join('data/fbtrain.csv'), delimiter=',', dtype=float)
data_test = np.genfromtxt(path.join('data/fbtest.csv'), delimiter=',', dtype=float)

X_train, y_train = np.array(data_train[:, 0:53]), np.array(data_train[:, 53])
X_test, y_test = np.array(data_test[:, 0:53]), np.array(data_test[:, 53])

# Normalize data
scaler = pp.StandardScaler().fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_test = scaler.transform(X_test)

# Instantiate DTC
clf = DecisionTreeRegressor(max_depth=3, random_state=10)
clf.fit(Xn_train, y_train)

grid = np.arange(1, 11, 1)  # try depths up to 10
params = {'max_depth': grid}

print("----------------------------\nDecision Tree Regressor")
clf_search = GridSearchCV(clf, params, scoring='neg_mean_squared_error')
clf_search.fit(Xn_train, y_train)

# Cross validation
score = clf_search.score(Xn_train, y_train)
print(f"Best param: {clf_search.best_params_}, cross val score: ", score*-1)

# Test error
clf = clf_search.best_estimator_

score = clf.score(Xn_test, y_test)
print("Test R2 coeff: ", score)
y_pred = clf.predict(Xn_test)
mse =  mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

print("----------------------------\nRandom forest Regressor")

# Random forest
clf = RandomForestRegressor(random_state=10)
clf_search = GridSearchCV(clf, params, scoring='neg_mean_squared_error')
clf_search.fit(Xn_train, y_train)

# Cross validation
score = clf_search.score(Xn_train, y_train)
print(f"Best param: {clf_search.best_params_}, cross val score: ", score*-1)

clf = clf_search.best_estimator_
score = clf.score(Xn_test, y_test)
print("Test R2 coeff: ", score)
y_pred = clf.predict(Xn_test)
mse =  mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

print("------------------------------\nFiltered h = 24")

filtered_train = np.array(data_train[data_train[:, 38] == 24])
filtered_test = np.array(data_test[data_test[:, 38] == 24])

X_train, y_train = np.array(filtered_train[:, 0:53]), np.array(filtered_train[:, 53])
X_test, y_test = np.array(filtered_test[:, 0:53]), np.array(filtered_test[:, 53])

scaler = pp.StandardScaler().fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_test = scaler.transform(X_test)

# Instantiate DTC
clf = DecisionTreeRegressor(max_depth=5, random_state=10)  # 5 best, 0.94
clf.fit(Xn_train, y_train)

print("----------------------------\nDecision Tree Regressor")
clf_search = GridSearchCV(clf, params, scoring='neg_mean_squared_error')
clf_search.fit(Xn_train, y_train)

# Cross validation
score = clf_search.score(Xn_train, y_train)
print(f"Best param: {clf_search.best_params_}, cross val score: ", score*-1)

# Test error
clf = clf_search.best_estimator_
score = clf.score(Xn_test, y_test)
print("Test R2 coeff: ", score)
y_pred = clf.predict(Xn_test)
mse =  mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

print("----------------------------\nRandom forest Regressor")

# Random forest
clf = RandomForestRegressor(random_state=10)
clf_search = GridSearchCV(clf, params, scoring='neg_mean_squared_error')
clf_search.fit(Xn_train, y_train)

# Cross validation
score = clf_search.score(Xn_train, y_train)
print(f"Best param: {clf_search.best_params_}, cross val score: ", score*-1)

clf = clf_search.best_estimator_
score = clf.score(Xn_test, y_test)
print("Test R2 coeff: ", score)
y_pred = clf.predict(Xn_test)
mse =  mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
