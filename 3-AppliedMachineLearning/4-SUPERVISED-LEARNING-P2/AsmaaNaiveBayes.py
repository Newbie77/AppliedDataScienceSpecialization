# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK4 - Supervised Learning P2 - Naive Bayes Classifiers


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_blobs, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.model_selection import train_test_split


# DATA PREPERATION

# prpare binary classificaiton data
X_C, y_C = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                               flip_y = 0.1, class_sep = 0.5, random_state=0)

# visualize data
plt.figure()
plt.title('Sample binary classification problem with two informative features')
plt.scatter(X_C[:, 0], X_C[:, 1], marker= 'o',
           c=y_C, s=50, cmap=cmap_bold)
plt.show()

# for more complex example

# create data
X_D, y_D = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
y_D = y_D % 2

# visualize data
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D[:,0], X_D[:,1], c=y_D, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

# real world dataset - cancer -
cancer = load_breast_cancer()
X_cancer, y_cancer = load_breast_cancer(return_X_y = True)

# ---------------------------------------------------------------------------------

# EX1
# split your data
X_train, X_test, y_train, y_test = train_test_split(X_C, y_C, random_state=0)

# fit the model with gaussian naive bayes 
nbclfier = GaussianNB().fit(X_train, y_train)

plot_class_regions_for_classifier(nbclfier, X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes classifier - Binary -')

# more complex example

# create data
X_D, y_D = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
y_D = y_D % 2

# visualize data
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D[:,0], X_D[:,1], c=y_D, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

# ---------------------------------------------------------------------------------

# EX2
# split data
X_train, X_test, y_train, y_test = train_test_split(X_D, y_D, random_state=0)

# fit the model with gaussian naive bayes classifier
nbclfier = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclfier, X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes classifier - Binary -')

# ---------------------------------------------------------------------------------

# test the classifier on real worls dataset - cancer dataset -

# split data
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

# fit the model
nbclfier = GaussianNB().fit(X_train, y_train)
print('Accuracy of GaussianNB classifier on training set:', nbclfier.score(X_train, y_train))
print('Accuracy of GaussianNB classifier on test set:', nbclfier.score(X_test, y_test))

