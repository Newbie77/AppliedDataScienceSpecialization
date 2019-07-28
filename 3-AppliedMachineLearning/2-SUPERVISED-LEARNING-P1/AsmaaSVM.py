# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - Support Vector Machines

# import libraries

from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot, plot_class_regions_for_classifier
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np


# LINEAR SVM FOR BINARY CLASSIFICATION

# create sample data
# for example: 2 features
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, flip_y=0.1, class_sep=0.5, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50)
plt.show()

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# create figure
fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))

# regularization parameter
c = 1.0

# create classifier and fit it
clf = SVC(kernel='linear', C=c).fit(X_train, y_train)

# plot results
title = 'Linear SVC, C = ' + str(c)
plot_class_regions_for_classifier_subplot(
    clf, X_train, y_train, None, None, title, subaxes)

# try different values of C parameter and note the changes
fig, subaxes = plt.subplots(1, 3, figsize=(8, 4))
for c, subplot in zip([0.00001, 1, 100], subaxes):

    # train the model
    clf = LinearSVC(C=c).fit(X_train, y_train)
    title = 'Linear SVC, C = ' + str(c)
    plot_class_regions_for_classifier_subplot(
        clf, X_train, y_train, None, None, title, subplot)

# ----------------------------------------------------------------------------

# MULTI-CLASS CLASSIFICATION WITH LINEAR SVM

# create data
# open data file
fruits = pd.read_table('fruit_data_with_colors.txt')

# specify feature names
feature_names = ['height', 'width', 'mass', 'color_score']

# split data to features and classes
X_f = fruits[feature_names]
y_f = fruits['fruit_label']

# specify class names
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

# use only 2 features
X_f2d = fruits[['height', 'width']]

# split data
X_train, X_test, y_train, y_test = train_test_split(X_f2d, y_f, random_state=0)

# create classifier and fit it
clf = LinearSVC(C=5, random_state=60).fit(X_train, y_train)

# print weights and intercept points
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)

# result visualization
# create figure
plt.figure(figsize=(6, 6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])

# plot data
# TODO: fix c issue şn scatter function
plt.scatter(X_f2d[['height']], X_f2d[['width']],
            cmap=cmap_fruits, edgecolor='black', alpha=.7)

x_range = np.linspace(-10, 15)

# plot each decision boundary
# multi class classification prblem will be solved by one-vs-all concept
for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b,
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_range, -(x_range * w[0] + b) / w[1], c=color, alpha=.8)

# plot results
plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()

# ----------------------------------------------------------------------------

# KERNELIZED SVM FOR CLASSIFICATION

# create sample data
X_k, y_k = make_blobs(n_samples=100, n_features=2,
                      centers=8, cluster_std=1.3, random_state=4)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_k, y_k, random_state=0)

# fit SVC model with the default kernel radial basis function (RBF) and plot the results
plot_class_regions_for_classifier(
    SVC().fit(X_train, y_train), X_train, y_train, None, None, 'SVC: RBF kernel')

# ----------------------------------------------------------------------------

# TYPES OF KERNELS

# polynomial kernel vs RBF
# Compare decision boundries with polynomial kernel, degree = 3
plot_class_regions_for_classifier(SVC(kernel='poly', degree=3).fit(X_train, y_train), X_train,
                                  y_train, None, None, 'SVC: Polynomial kernel, degree = 3')

# the effect of gamma parameter on SVM
fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

# plot and note corresponding overfitting
for gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
    clf = SVC(kernel='rbf', gamma=gamma).fit(X_train, y_train)
    title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(
        gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                              None, None, title, subplot)

# ----------------------------------------------------------------------------

# GAMMA AND C PARAMETER

# get all together
# combining C and gamma parameters
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

# try different values for both gamma and c and observe the results
for gamma, this_axis in zip([0.01, 1, 5], subaxes):
    for c, subplot in zip([0.1, 1, 15, 250], this_axis):
        # set title
        title = 'gamma = {:.2f}, C = {:.2f}'.format(gamma, c)

        # train the model by specified c and gamma
        clf = SVC(kernel='rbf', gamma=gamma, C=c).fit(X_train, y_train)

        # plot results
        plot_class_regions_for_classifier_subplot(
            clf, X_train, y_train, X_test, y_test, title, subplot)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)