# author: Asmaa ~ 2019
# Coursera: Applied Plotting, Charting and Data Representation
# WEEK4 - Applied Visualizations - Pandas Visualization

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# PANDAS VISUALIZATION

# create a data frame
data = pd.DataFrame(
    {
        "d1": np.random.randn(200),
        "d2": np.random.randn(200),
        "d3": np.random.randn(200),
    },
    index=pd.date_range("28-9-2013", periods=200),
)

print(data.head())

# plot data
# graphs oof pandas: different colors, legend,
data.plot()

# more options
# two axes, discrete data
# plot(horizontal axis, vertical axis, kind=plot type)
data.plot("d1", "d2", kind="scatter", c="r")

# kind values:
# 'line' : line plot (default)
"""
'bar' : vertical bar plot
'barh' : horizontal bar plot
'hist' : histogram
'box' : boxplot
'kde' : Kernel Density Estimation plot
'density' : same as 'kde'
'area' : area plot
'pie' : pie plot
'scatter' : scatter plot
'hexbin' : hexbin plot
"""

# more pltting options
# create a scatter plot of columns 'd1' and 'd2',
# with changing color (d2) and size (s) based on column 'd3'

axes = data.plot.scatter("d1", "d3", c="d2", s=data["d2"], colormap="viridis")

# set equal paritions
axes.set_aspect("equal")

# plot boxes
data.plot.box()


# plot histograms
data.plot.hist()

# plot probability density functions (pdf)
data.plot.kde()

# pandas.tools.plotting
iris = pd.read_csv("iris.csv")
print(iris.head())

# plt scatter_matrix, gives info about the relations between each column in the data set
pd.tools.plotting.scatter_matrix(iris)

# --------------------------------------------------------------

# SEABORN VISUALIZATION

# create data
d1 = pd.Series(np.random.normal(0, 3, 1000), name="d1")
d2 = pd.Series(2 * d1 + np.random.normal(10, 5, 1000), name="d2")
plt.figure()
plt.hist(d1, label="d1")
plt.hist(d2, label="d2")
plt.legend()

# plot a kernel density estimation over a stacked barchart
plt.figure()
plt.hist([d1, d2], histtype="barstacked", normed=True)
d3 = np.concatenate((d1, d2))
sns.kdeplot(d3)

# joint plot
sns.distplot(d3)
sns.jointplot(d1, d2)

# set aspect ratio
grid = sns.jointplot(d1, d2, alpha=0.4)
grid.ax_joint.set_aspect("equal")

# set visualizing style
sns.jointplot(d1, d2, kind="hex")

# TODO: search for more plotting options and functionalities in seaborn

