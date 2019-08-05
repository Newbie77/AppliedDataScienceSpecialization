# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK3 - Evaluation - Regression Evaluation Metrics

# import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# REGRESSION EVALUATION METRICS

# load data
diabete = datasets.load_diabetes()

X = diabete.data[:, None, 6]
y = diabete.target


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit linear regression model
linrr = LinearRegression().fit(X_train, y_train)

# fit dummy model
linrr_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)

# predict values using new models
y_p = linrr.predict(X_test)
y_p_dummy = linrr_dummy_mean.predict(X_test)


# print evaluation metrics
print('Linear model, coefficients: ', linrr.coef_)
print("Mean squared error (dummy):", (mean_squared_error(y_test, y_p_dummy)))
print("Mean squared error (linear model):", mean_squared_error(y_test, y_p))
print("r2_score (dummy):", r2_score(y_test, y_p_dummy))
print("r2_score (linear model):", r2_score(y_test, y_p))

# plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_p, color='green', linewidth=2)
plt.plot(X_test, y_p_dummy, color='red', linestyle = 'dashed', linewidth=2, label = 'dummy')

plt.show()
