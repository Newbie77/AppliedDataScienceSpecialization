# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - Cross Validation


# import libraries
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# CROSS VALIDATION

# PREPARE DATA
# open data file
fruits = pd.read_table('fruit_data_with_colors.txt')

# specify feature names 
feature_names = ['height', 'width', 'mass', 'color_score']

# split inputs and output
X_fruits = fruits[feature_names]
y_fruits = fruits['fruit_label']

# use only 2 features
X_f2d = fruits[['height', 'width']]

# PREPARE CLASSIFIER

# use knn with k=5
clfier = KNeighborsClassifier(n_neighbors = 5)

# to matrix
X = X_f2d.as_matrix()
y = y_fruits.as_matrix()

# do cross validation and get the scores
cv_scores = cross_val_score(clfier, X, y)

# print scores
print('Cross-validation scores (3-fold):', cv_scores)

# print mean of scores
print('Mean cross-validation score (3-fold): ', np.mean(cv_scores))

# -------------------------------------------------------------------------

# VALIDATION CURVES

# create parameter range
param_range = np.logspace(-3, 3, 4)

# calculate validation scores 3 fold
train_scores, test_scores = validation_curve(SVC(), X, y, param_name='gamma', param_range=param_range, cv=3)

# print results
print(train_scores)
print(test_scores)

# visualize results
plt.figure()

# calculate means and standard deviation for each train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# prepare the figure
plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label='Training score',
            color='orange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='orange', lw=lw)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
            color='black', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='black', lw=lw)

plt.legend(loc='best')
plt.show()
