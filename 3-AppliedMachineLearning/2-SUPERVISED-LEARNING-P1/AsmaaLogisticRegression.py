# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - Logistic Regression

# import libraries
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import pandas as pd
from sklearn.model_selection import train_test_split

# DATA PREPARATION

# open data file
fruits = pd.read_table('fruit_data_with_colors.txt')

# specify features
features = ['height', 'width', 'mass', 'color_score']

# split data to X and y
X = fruits[features]
y = fruits['fruit_label']

# get only 2 features
X2d = fruits[['height', 'width']]

# create figure
fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))

# convert it into  a binary classification problem, is it an apple or not?
# apples vs everything else
y_apple = y == 1  


# CREATING THE MODEL

# split dataset
X_train, X_test, y_train, y_test = (
train_test_split(X2d.as_matrix(), y_apple.as_matrix(), random_state = 0))

# fit the model
clf = LogisticRegression(C=100).fit(X_train, y_train)

# plot the results
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
                                         None, 'Logistic regression: Apple vs others',subaxes)
subaxes.set_xlabel('height')
subaxes.set_ylabel('width')

# use the model for other values
h = 6
w = 8
print('A fruit with height {} and width {} is predicted to be: {}'
     .format(h,w, ['not an apple', 'an apple'][clf.predict([[h,w]])[0]]))

# find accuracy on train and test dataset 
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# --------------------------------------------------------------------------------------

# REGULARIZATION PARAMETER

# effect of C value (regularization parameter)
# redesign figure
fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))

# try different C values
for c, subplot in zip([0.1, 1, 100], subaxes):
    # fit the model
    clf = LogisticRegression(C=c).fit(X_train, y_train)
    
    # set title
    title ='Logistic regression (apple vs rest), C = ' +str(c)
    
    # plot
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subplot)

