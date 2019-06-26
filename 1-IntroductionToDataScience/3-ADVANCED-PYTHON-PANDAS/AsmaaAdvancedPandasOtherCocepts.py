# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK3 Advanced Pandas - Scales, Pivot Table and Dates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SCALES
# create a data farme
# Nominal Scale
grades = pd.DataFrame(['AA', 'BA', 'BB', 'CB', 'CC', 'DC', 'DD', 'FD', 'FF'], index=[
                      'perfect', 'excellent', 'excellent', 'good', 'good', 'ok', 'ok',
                      'failed', 'failed']).rename(columns={0: 'letter'})


# convert into Ordinal Scale
ordered_grades = grades['letter'].astype('category', categories=[
                                         'FF', 'FD', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA'], ordered=True)

# apply boolean masking
course = pd.DataFrame(['FF', 'DC', 'AA'])
print(course < 'CC')


# PIVOT TABLES
cars = pd.read_csv('cars.csv')
print(cars.shape)

# returns a table has YEAR field (distinct values) as row names and
# Make field as column names and the values are the average of kW
# field for corresponding Make and YEAR values (Liked it ^_^)
pivot = cars.pivot_table(values='(kW)', index='YEAR',
                         columns='Make', aggfunc=np.mean)
print(pivot)

# --------------------------------------------------------
# DATES

# common functions
# get time and date
ts = pd.Timestamp('7/11/1998 12:15AM')
print(ts)

# get month or day
m = pd.Period('11/1998')
d = pd.Period('7/11/1998')
print(m, d)

# DateTimeIndex - use dates as indices
s = pd.Series(['asmaa', 'esma'], [pd.Timestamp(
    '7/11/1998 12AM'), pd.Timestamp('7/11/1998 12PM')])
print(s)


# date formatting (Liked it :D)
# in default yyyy-dd-mm
dates = ['7/11/1998', '26.6.2019', 'June 5 2015']
print(pd.to_datetime(dates))

# yyyy-mm-dd
print(pd.to_datetime(dates, dayfirst=True))

# Time Deltas (time differences) (LIKED it :p)
delta = pd.Timestamp('26/6/2019')-pd.Timestamp('7/11/1998')
print(delta)

# date range (start date, number of terms, length of term)
dr = pd.date_range('28/9/2013', periods=7, freq='2M')
print(dr)

# use date range as an index for data frame
data = pd.DataFrame({'Num 1':  np.random.randint(0, 30, 7).cumsum(),
                     'Count 2': np.random.randint(0, 60, 7)}, index=dr)


print(data)

# querying
print(data['2014'])
print(data['2014-05'])


# change frequency of dates
print(data.asfreq('1M', method='ffill'))

data.plot()
