# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK4 - Supervised Learning P2 - Neural Networks


# import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_regression, load_breast_cancer
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier, MLPRegressor
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot, plot_class_regions_for_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ACTIVATION FUNCTIONS VISUALIZATION

x_range = np.linspace(-2, 2, 200)

plt.figure()

# Relu
plt.plot(x_range, np.maximum(x_range, 0), label = 'relu')

# Hyperbolic tangant
plt.plot(x_range, np.tanh(x_range), label = 'tanh')

# Sigmoid function
plt.plot(x_range, 1 / (1 + np.exp(-x_range)), label = 'logistic')

# set plotting options
plt.legend()
plt.title('Neural network activation functions')
plt.xlabel('Input value (x)')
plt.ylabel('Activation function output')
plt.show()

# ----------------------------------------------------------------

# DATA PREPARATION

# classification data
# create data
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X, y = make_blobs(n_samples = 100, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
y = y % 2

# visualize data
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X[:,0], X[:,1], c=y, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

# regression data
X_R, y_R = make_regression(n_samples = 100, n_features=1, n_informative=1, bias = 150.0, noise = 30, random_state=0)
plt.scatter(X_R, y_R, marker= 'o', s=50)
plt.show()

# prepare cancer dataset
cancer = load_breast_cancer()
X_cancer, y_cancer = load_breast_cancer(return_X_y = True)

# ----------------------------------------------------------------

# CLASSIFICATION WITH NN

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))


# train the model with different number of units in the hidden layer 1 unit, 10 units, 100 units
for units, axis in zip([1, 10, 100], subaxes):
    nnclfier = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs', random_state = 0).fit(X_train, y_train)
    
    title = 'Neural net classifier, 1 layer, {} units'.format(units)
    
    plot_class_regions_for_classifier_subplot(nnclfier, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()
    
# RESULTS:
# 1 unit    ==> underfitting
# 10 units  ==> little better
# 100 units ==> tends to overfitting :/

# re-do operations for two hidden layers with 10 units in each one
#nnclfier = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs', random_state = 0).fit(X_train, y_train)
fig, subaxes = plt.subplots(3, 1, figsize=(6,18))
for units, axis in zip([1, 10, 100], subaxes):
    nnclfier = MLPClassifier(hidden_layer_sizes = [units, units], solver='lbfgs', random_state = 0).fit(X_train, y_train)
    
    title = 'Neural net classifier, 2 layer, {} units'.format(units)
    
    plot_class_regions_for_classifier_subplot(nnclfier, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()

# ----------------------------------------------------------------

# REGULARIZATION PARAMETER

# The effect of regularization parameter (alpha)
fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))

for alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclfier = MLPClassifier(solver='lbfgs', activation = 'tanh', alpha = alpha,
                         hidden_layer_sizes = [100, 100], random_state = 0).fit(X_train, y_train)
    
    title = 'NN classifier, alpha = {:.3f} '.format(alpha)
    
    plot_class_regions_for_classifier_subplot(nnclfier, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()

# The effect of different choices of activation function
fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for act, axis in zip(['logistic', 'tanh', 'relu'], subaxes):
    nnclfier = MLPClassifier(solver='lbfgs', activation = act,
                         alpha = 0.1, hidden_layer_sizes = [10, 10], random_state = 0).fit(X_train, y_train)
    
    title = 'NN classifier, 2 layers 10/10, {} activation function'.format(act)
    
    plot_class_regions_for_classifier_subplot(nnclfier, X_train, y_train, X_test, y_test, title, axis)

# ----------------------------------------------------------------

# REGRESSION WITH NN
# split data
X_train, X_test, y_train, y_test = train_test_split(X_R[0::5], y_R[0::5], random_state = 0)
X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)


# fit the model for different activation functions and different values of alpha
fig, subaxes = plt.subplots(2, 3, figsize=(11,8), dpi=70)
for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        mlpregressor = MLPRegressor(hidden_layer_sizes = [100,100], activation = thisactivation,
                             alpha = thisalpha, solver = 'lbfgs').fit(X_train, y_train)
        y_predict_output = mlpregressor.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output,
                     '^', markersize = 10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('MLP regression\nalpha={}, activation={})'
                          .format(thisalpha, thisactivation))
        plt.tight_layout()

# ----------------------------------------------------------------

# apply NN on real world dataset - cancer -
scaler = MinMaxScaler()

# split data
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit the model
clfier = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0, random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

# print the results
print('Accuracy of NN classifier on cancer training set:', (clfier.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on cancer test set:',clfier.score(X_test_scaled, y_test))