

# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - Datasets

from sklearn.datasets import make_friedman1
from sklearn.datasets import make_classification, make_blobs, make_regression
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# generate a random regression problem

# make_regression(number of samples, number of features, number of informative features, ...)
# n_informative:  number of features used to build the linear model used to generate the output

X_R1, y_R1 = make_regression(n_samples=100, n_features=1,
                             n_informative=1, bias=150.0,
                             noise=30, random_state=0)

# plot values
plt.figure()
plt.title('Random sample regression problem')
plt.scatter(X_R1, y_R1, marker='o', s=50)
plt.show()

# -------------------------------------------------------------------------------

# more complex regression example

# make friedman: Inputs X are independent features uniformly distributed on the
# interval [0, 1]. The output y is created according to the formula:
# y = 10 sin(π x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + e
# make_friedman1(number of samples, number of features, random number generation for dataset noise)

X_F1, y_F1 = make_friedman1(n_samples=100,
                            n_features=7, random_state=0)

# plot data
plt.figure()
plt.title('Complex regression problem')
plt.scatter(X_F1[:, 2], y_F1, marker='o', s=50)
plt.show()

# -------------------------------------------------------------------------------

# classification example
# make random sample
# binary classification
X_C2, y_C2 = make_classification(n_samples=100, n_features=2,
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0.1,
                                 class_sep=0.5, random_state=0, n_classes=2)

# plot data
plt.figure()
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF', '#000000'])
plt.title('Sample binary classification')
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
            marker='o', s=50, cmap=cmap_bold)
plt.show()

# -------------------------------------------------------------------------------

# classification example

# make random sample
# multi-class classification (4)
# class_sep parameter defines how data will be seperated and easy to classify (0.1 is too bad!)
X_C2, y_C2 = make_classification(n_samples=100, n_features=2,
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0.1,
                                 class_sep=0.1, random_state=0, n_classes=4)

# plot data
plt.figure()
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF', '#000000'])
plt.title('Sample binary classification')
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
            marker='o', s=50, cmap=cmap_bold)
plt.show()

# -------------------------------------------------------------------------------

# more complex classification (binary)

# classes that are not linearly separable
# make_blobs: generates isotropic Gaussian blobs for clustering.

X_D2, y_D2 = make_blobs(n_samples=100, n_features=2, centers=4,
                        cluster_std=1.3, random_state=0)

# break the linearity (Not sure)
y_D2 = y_D2 % 2

# plot data
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:, 0], X_D2[:, 1], c=y_D2,
            marker='o', s=50, cmap=cmap_bold)
plt.show()
