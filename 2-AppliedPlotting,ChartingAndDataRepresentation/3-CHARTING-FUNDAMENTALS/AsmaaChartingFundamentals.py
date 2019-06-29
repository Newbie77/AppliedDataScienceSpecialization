# author: Asmaa ~ 2019
# Coursera: Applied Plotting, Charting and Data Representation
# WEEK3 - Charting Fundamentals: Subplots - Histograms - Box plots - Heatmaps - Animations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as pan

# SUBPOLTS
# create new figure
plt.figure()

# create subplots subplot(row_num, column_num, current_axis)
# get first area
plt.subplot(1, 3, 1)

# plot at first area
data = np.array([1, 3, 7, 9, 11])
plt.plot(data)

# get second area of the figure
plt.subplot(1, 3, 2)

# plot at second area
# you can plot multiple charts :D
data2 = np.array([2, 4, 6, 8])
plt.plot(data2, ".-m")
plt.plot(data, ".-c")

# get third area
plt.subplot(1, 3, 3)

# plot at third area
data3 = np.array([1, 4, 9, 16])
plt.plot(data3, "*-r")

# creating grids
# a 2*2 grid
# subplot(row_num, column_num, invidual or common axis labels?)
fig, ((sub1, sub2), (sub3, sub4)) = plt.subplots(2, 2, sharex=True, sharey=True)

# plot at every part
# Liked it :D
sub1.plot(data)
sub2.plot(data2)
sub3.plot(data3)
sub4.plot(data)

# --------------------------------------------------------------------

# HISTOGRAMS
# definition: a diagram consisting of rectangles whose area
# is proportional to the FREQUENCY of a variable

# plot single histogram
plt.figure()
hist_data = np.random.randint(50, size=100)
plt.hist(hist_data, bins=25)

# plot multiple histograms

# create cavnas
fig, ((hist1, hist2), (hist3, hist4)) = plt.subplots(2, 2, sharex=False, sharey=False)

# assign subplots to an array to iterate over
hists = [hist1, hist2, hist3, hist4]

# iterate and plot
for hist in hists:
    data = np.random.normal(loc=1, scale=2, size=500)
    hist.hist(data, bins=25)

# use scatter for random data :D :p
plt.figure()
x = np.random.rand(1000)
y = np.random.rand(1000)
plt.scatter(x, y, c="pink")


# --------------------------------------------------------------------

# BOX PLOTS

# create some data
ndata = np.random.normal(loc=0.0, scale=1.0, size=10000)
rdata = np.random.random(size=10000)
gdata = np.random.gamma(2, size=10000)

# create data frame
df = pd.DataFrame({"normal": ndata, "random": rdata, "gama": gdata})

#  get info about data
print(df.describe())

# now create box plot
plt.figure()
plt.boxplot([df["normal"], df["random"], df["gama"]], whis="range")

# --------------------------------------------------------------------

# HEATMAPS

# create a new figure
plt.figure()

# create some data
outputs = np.random.normal(loc=1, scale=2, size=10000)
inputs = np.random.random(size=10000)

# plot data
plt.hist2d(inputs, outputs, bins=50)

# show color bar
plt.colorbar()

# --------------------------------------------------------------------

# ANIMATION
x = np.random.randn(100)

# create function that will be invoked for pltting
def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n:
        a.event_source.stop()
    # clear
    plt.cla()
    # create bins
    bins = np.arange(-4, 4, 0.5)
    # plot data
    plt.hist(x[:curr], bins=bins)
    # set axes boundaries
    plt.axis([-4, 4, 0, 30])
    # set labels
    plt.gca().set_title("Sampling the Normal Distribution")
    plt.gca().set_ylabel("Frequency")
    plt.gca().set_xlabel("Value")
    plt.annotate("n = {}".format(curr), [3, 27])


# create new figure
fig = plt.figure()
# apply animation
a = pan.FuncAnimation(fig, update, interval=100)

