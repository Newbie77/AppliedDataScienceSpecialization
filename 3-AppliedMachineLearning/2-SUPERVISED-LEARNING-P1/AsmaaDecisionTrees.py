# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - Decision Trees


# import libraries
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree, plot_feature_importances
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# BASIC MODEL

# load data
iris_data = load_iris()

# split data
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, random_state = 3)

# fit the model
clfier = DecisionTreeClassifier().fit(X_train, y_train)

# print truth metrics
print('Accuracy on training set: ', clfier.score(X_train, y_train))
print('Accuracy on test set: ', clfier.score(X_test, y_test))

# -------------------------------------------------------------------------------------------

# AVOIDING OVERFITTING by setting max depth of the tree
# fit the model again
clfier2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

# print truth metrics
print('Accuracy on training set:', format(clfier2.score(X_train, y_train)))
print('Accuracy on test set:', clfier2.score(X_test, y_test))

# VISUALAZATION OF DECİSİON TREES
plot_decision_tree(clfier, iris_data.feature_names, iris_data.target_names)
plot_decision_tree(clfier2, iris_data.feature_names, iris_data.target_names)

# -------------------------------------------------------------------------------------------

# DECISION TREES ON REAL-WORLD DATASET

# load data
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

# fit the model
cancer_clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8, random_state = 0).fit(X_train, y_train)

# visualize decision tree
plot_decision_tree(cancer_clf, cancer.feature_names, cancer.target_names)

# print truth metrics
print('Accuracy of on training set:', cancer_clf.score(X_train, y_train))
print('Accuracy of on test set:', cancer_clf.score(X_test, y_test))


plt.figure(figsize=(10,6),dpi=80)
plot_feature_importances(cancer_clf, cancer.feature_names)
plt.tight_layout()

plt.show()
