# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK1 Python Fundamentals - Numpy

import numpy as np

# create array
arr = np.array([11, 22, 33, 44, 55, 66, 77])
print(arr)

# create 2D array and get size
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix.shape)

# list-range from 1 to 14 increment by 2
lrange = np.arange(1, 15, 2)
print(lrange)

#  list-range 10 values starts by 0 end by 4
r = np.linspace(0, 4, 10)
print(r)

# ones
ones = np.ones((3, 2), float)
print(ones)

# zeros
zeros = np.zeros((3, 2), int)
print(zeros)

# random array randint(min, max+1, (row, column))
rand = np.random.randint(0, 10, (3, 3))
print(rand)
