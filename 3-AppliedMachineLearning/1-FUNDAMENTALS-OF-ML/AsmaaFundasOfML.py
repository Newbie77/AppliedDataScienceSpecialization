# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK1 - Fundamentals of Machine Learning - Intro to SciKit Learn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

# open data set file with Pandas
fruits = pd.read_csv("fruit_data_with_colors.txt", "\t")
print(fruits.head())

# make a label for each fruit type as a number
num_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(num_fruit_name)

# split data set

# feature matrix
X = fruits[["mass", "width", "height"]]

# output vector
y = fruits["fruit_label"]

# solit records in data set ==> training set 75%, testing set 25%
# random_satate: seed randomise splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# then we can plot data set to inspect data and relations between features
# we can use scatter matrix to see the relations between features
cmap = cm.get_cmap("gnuplot")
scatter = pd.scatter_matrix(
    X_train,
    c=y_train,
    marker="o",
    s=40,
    hist_kwds={"bins": 15},
    figsize=(9, 9),
    cmap=cmap,
)

# and we can plot data in 3D space for better inspection

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X_train["width"],
    X_train["height"],
    c=y_train,
    marker="o",
    s=100,
)
ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_zlabel("color_score")
plt.show()

# now we can apply KNN algorithm using sklearn lib function
# we will apply it for 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# start training the classifier
knn.fit(X_train, y_train)

# test the classifier
knn.score(X_test, y_test)

# use the classifier for more examples that you want
# 10g, width 3 cm, height 5.1 cm
predict = knn.predict([[10, 3, 5.1]])
print(predict)
num_fruit_name[predict[0]]

# one more exapmle
# 80g, width 6 cm, height 4.5 cm
predict = knn.predict([[80, 6, 4.5]])
num_fruit_name[predict[0]]

# plot accuracy

k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);