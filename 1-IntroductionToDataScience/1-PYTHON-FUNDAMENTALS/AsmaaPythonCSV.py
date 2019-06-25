# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK1 Python Fundamentals - CSV operations

import csv

with open('mpg.csv') as my_csv:
    data = list(csv.DictReader(my_csv))

print(data[:4])

print(len(data))

print(data[0].keys())

# find average of cty
avg = sum(int(item['cty']) for item in data) / len(data)
print(avg)

# distinct values of year
year = set(item['year'] for item in data)
print(year)

# distinct values of class
classes = set(item['class'] for item in data)
print(classes)

# average hwy for each class
avg_hwy = []
for c in classes:
    avg_hwy.append(sum(int(item['hwy']) for item in data) /
                   sum(item['class'] == c for item in data))

print(avg_hwy)

# store as a dictionary
avg_dict = dict(zip(classes, avg_hwy))
print(avg_dict)
