# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK4 - Supervised Learning P2 - Random Forests & GBDT

# import libraries

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

# DATA PREPARATION

# create data
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X, y = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
y = y % 2

# visualize data
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X[:,0], X[:,1], c=y, marker = 'o', s=50, cmap=cmap_bold)
plt.show()

# create real world data set
cancer = load_breast_cancer()
X_cancer, y_cancer = load_breast_cancer(return_X_y = True)

# ----------------------------------------------------------------------------

# RANDOM FOREST CLASSIFIERS

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# fit the model
clfier = RandomForestClassifier().fit(X_train, y_train)

# visualize results
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

title = 'Random Forest Classifier'
plot_class_regions_for_classifier_subplot(clfier, X_train, y_train, X_test, y_test, title, subaxes)

plt.show()

# use random forest classifier on real world dataset - cancer -
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

# fit the model
clfier = RandomForestClassifier(max_features = 8, random_state = 0).fit(X_train, y_train)

# print the results
print('Accuracy of RF on training set:', clfier.score(X_train, y_train))
print('Accuracy of RF on test set:', clfier.score(X_test, y_test))

# ----------------------------------------------------------------------------

# GRADIENT BOOSTED DECISION TREES

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# fit the model
clfier = GradientBoostingClassifier().fit(X_train, y_train)

# visualize the results
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))
title = 'GBDT results'
plot_class_regions_for_classifier_subplot(clfier, X_train, y_train, X_test, y_test, title, subaxes)

plt.show()

# use GBDT on real world dataset - cancer -
# split the data
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

# fit the model
# default learning rate: 0.1, max_depth: 3
clfier = GradientBoostingClassifier(random_state = 0).fit(X_train, y_train)

# print the results
print('Accuracy of GBDT on training set:', clfier.score(X_train, y_train))
print('Accuracy of GBDT on test set:', clfier.score(X_test, y_test))


# change the parameters to avoid underfitting or overfitting
clfier = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0).fit(X_train, y_train)

# print the rsults and compare
print('Accuracy of GBDT on training set:', clfier.score(X_train, y_train))
print('Accuracy of GBDT on test set:', clfier.score(X_test, y_test))
