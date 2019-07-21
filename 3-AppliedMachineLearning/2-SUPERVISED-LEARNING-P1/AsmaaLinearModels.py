# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK2 - Supervised Learning - Linear models for regression


# import libraries
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import PolynomialFeatures


# LINEAR REGRESSION
# create sample data
X, y = make_regression(n_samples=100, n_features=1, n_informative=1, bias=150.0, noise=30, random_state=0)

# split data 75% for training --- 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# plot sample data
plt.scatter(X, y, marker='o', s=50)

# train the model
reg = LinearRegression().fit(X_train, y_train)

# print coefficients (w value) of the trained model
print('linear model coeff (w):', reg.coef_)

# print b value (intercept with y)
print('linear model intercept (b):', reg.intercept_)

# ending with _ like (coef_) means that this value evaluted from training set only

# print score on train data
print('R-squared score (training):', reg.score(X_train, y_train))

# print score on test data
print('R-squared score (test):', reg.score(X_test, y_test))

# plot model

# plot data
plt.scatter(X, y, marker='o')

# plot model line
plt.plot(X, reg.coef_ * X + reg.intercept_, 'r-')

# -----------------------------------------------------------------------------

# RIDGE REGRESSION
# Linear regression with regularization parameter

# create separated data set
Xr, yr = make_regression(n_samples=200, n_features=1, n_informative=1, bias=0,
                         noise=100, random_state=10)
plt.scatter(Xr, yr, marker='o')

# split data set
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, random_state=0)

# train ridge model with alpha = 20
ridge = Ridge(alpha=20.0).fit(Xr_train, yr_train)


# print model parameters
# print coefficients (w value) of the trained model
print('linear model coeff (w):', ridge.coef_)

# print b value (intercept with y)
print('linear model intercept (b):', ridge.intercept_)

# ending with _ like (coef_) means that this value evaluted from training set only

# print score on train data
print('R-squared score (training):', ridge.score(Xr_train, yr_train))

# print score on test data
print('R-squared score (test):', ridge.score(Xr_test, yr_test))

# declare a scaler
scaler = MinMaxScaler()

# store scaled data
Xr_train_scaled = scaler.fit_transform(Xr_train)
Xr_test_scaled = scaler.transform(Xr_test)


# repeat same operations


# train ridge model with alpha = 20
ridge = Ridge(alpha=20.0).fit(Xr_train_scaled, yr_train)


# print model parameters
# print coefficients (w value) of the trained model
print('linear model coeff (w):', ridge.coef_)

# print b value (intercept with y)
print('linear model intercept (b):', ridge.intercept_)

# ending with _ like (coef_) means that this value evaluted from training set only

# print score on train data
print('R-squared score (training):', ridge.score(Xr_train_scaled, yr_train))

# print score on test data
print('R-squared score (test):', ridge.score(Xr_test_scaled, yr_test))

# -----------------------------------------------------------------------------

# POLYNOMIAL REGRESSION
# create data
X_P, y_P = make_friedman1(n_samples=100, n_features=7, random_state=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_P, y_P, random_state=0)

# use linear regression model
lineareg = LinearRegression().fit(X_train, y_train)

# print model info
print('linear model coeff (w):', lineareg.coef_)
print('linear model intercept (b):', lineareg.intercept_)
print('R-squared score (training):', lineareg.score(X_train, y_train))
print('R-squared score (test):', lineareg.score(X_test, y_test))

# add polynomial features degree = 2
# re-create data
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_P)

# re-split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_P,
                                                    random_state=0)

# re-use linear regression
lineareg = LinearRegression().fit(X_train, y_train)

# print new model info
print('(poly deg 2) linear model coeff (w):', lineareg.coef_)
print('(poly deg 2) linear model intercept (b):', lineareg.intercept_)
print('(poly deg 2) R-squared score (training):',
      lineareg.score(X_train, y_train))
print('(poly deg 2) R-squared score (test):', lineareg.score(X_test, y_test))
# NOW we have better values !!! :D


# NOTE: polynomial features often lead to overfitting, so we have to combine polynomial
# features and regulatization to avoid overfitting

# re-create data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_P,
                                                    random_state=0)

# use ridge instead of normal linear regression
lineareg = Ridge().fit(X_train, y_train)

print('(poly deg 2 + ridge) linear model coeff (w):', lineareg.coef_)
print('(poly deg 2 + ridge) linear model intercept (b):', lineareg.intercept_)
print('(poly deg 2 + ridge) R-squared score (training):',
      lineareg.score(X_train, y_train))
print('(poly deg 2 + ridge) R-squared score (test):',
      lineareg.score(X_test, y_test))

# now we have lower score on training set but higher score on test set
