# author: Asmaa ~ 2019
# Coursera: Applied Plotting, Charting and Data Representation
# WEEK2 - Basic Charting: basic plotting, scatterplots, line plot, bar chart

import matplotlib.pyplot as plt
import numpy as np


# plot sibgle point
# plot(x, y, render_style)
plt.plot(3, 4, "*")

# create a figure and plot points
plt.figure()
plt.plot(5, 6, "*")
plt.plot(3, 2, "o")
plt.plot(4, 1, ".")

# get current axes
ax = plt.gca()

# specify axes boundries axis(min_x, max_x, min_y, max_y)
ax.axis([0, 7, 0, 13])

# get all elements of the chart
elem = ax.get_children()

# plotting arrays
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x * 4

plt.figure()
plt.scatter(x, y)

# coloring data
# LIKED IT :D
plt.figure()

# declare list of colors
colors = ["blue"] * (len(x) - 1)
colors.append("purple")
# scatter(data, data, size of points, color list, label)
plt.scatter(x, y, s=100, c=colors)

# plot as multiple parts
plt.figure()
plt.scatter(x[:4], y[:4], s=50, c="pink", label="first part")
plt.scatter(x[4:], y[4:], s=50, c="green", label="second part")

# label axes and graph now
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("X and Y graph")

# add legend due to labels in scatter function
plt.legend()

# legend position legend(location due to geometric partition, frame?, title)
plt.legend(loc=4, frameon=True, title="description")

# LINE PLOT
# write - before or after the symbol

plt.figure()
data = list(zip([1, 2, 3, 4, 5, 6], [1, 4, 9, 16, 25, 36]))
plt.plot(data, "*-")

# specify color with symbol r=red, y=yellow, .....
data = list(zip([1, 2, 3, 4, 5, 6], [1, 4, 9, 16, 25, 36]))
plt.plot(data, "*-y")

# BAR CHART
# bar(input, output, width of each bar)
inputs = [0, 2, 4, 6, 8, 10]
outputs = [1, 3, 7, 15, 8, 5]
plt.figure()
plt.bar(inputs, outputs, width=0.5)

inputs2 = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5]
outputs2 = [15, 3, 1, 5, 3, 4]
plt.bar(inputs2, outputs2, width=0.4, color="pink")

# horizontal bar
plt.figure()
plt.barh(inputs, outputs, height=0.3, color="red")

