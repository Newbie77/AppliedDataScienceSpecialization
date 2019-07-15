# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - K-Nearest Neighbors


from adspy_shared_utilities import plot_two_class_knn
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# CLASSIFICATION

# create data
X, y = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)

# split data, default ratio: 0.25 * 0.75
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# plot data with different K value and compare th results
plot_two_class_knn(X_train, y_train, 1, 'uniform', X_test, y_test)
plot_two_class_knn(X_train, y_train, 3, 'uniform', X_test, y_test)
plot_two_class_knn(X_train, y_train, 11, 'uniform', X_test, y_test)

# conclusion: K value can differnetiate between 1 and m_train (when m_train is number of records in tain set)
# K=1 ==> overfitting, high accuracy on train set + low accuracy on test set
# K=m_train ==> underfitting, low accuracy on both train and test sets (one output: class that has high frequency)
# the best K is the value that corresponds to the highest accuracy on both train and test sets
# K must satisfy 1 < K < m_train

# ---------------------------------------------------------------------------------------

# REGRESSION

# create data
X_R, y_R = make_regression(n_samples=100, n_features=1,
                             n_informative=1, bias=150.0,
                             noise=30, random_state=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_R, y_R, random_state = 0)

# fit the model with K=5
reg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

# test the results
# print(reg.predict(X_test))
print('R-squared test score: {:.3f}'
     .format(reg.score(X_test, y_test)))

# Output: R-squared test score: 0.425

# try other values K=1
reg = KNeighborsRegressor(n_neighbors = 1).fit(X_train, y_train)
print('R-squared test score: {:.3f}'
     .format(reg.score(X_test, y_test)))
# Output: R-squared test score: 0.155
# underfitting

# try K=60
reg = KNeighborsRegressor(n_neighbors = 60).fit(X_train, y_train)
print('R-squared test score: {:.3f}'
     .format(reg.score(X_test, y_test)))

# Output: R-squared test score: 0.322

