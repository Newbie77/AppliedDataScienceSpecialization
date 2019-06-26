# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK2 Basic Data Processing with Pandas - Series and Data Frames, Reading CSV and Data Prcessing

import pandas as ps
import numpy as np

# SERIES
# declare a series
names = ['asmaa', 'esma']
series = ps.Series(names)
print(series)

# declare a series from a dictionary
user = {'name': 'asmaa', 'surname': 'mirkhan', 'city': 'istanbul'}
series = ps.Series(user)
print(series)
# Note: output is sorted alphabetically by indices

# another way for declaration
series = ps.Series(['asmaa', 'mirkhan'], index=['name', 'surname'])
print(series)

# querying
# by numerical index
print(series.iloc[1])
# by key
print(series.loc['name'])

# vectorization
ss = ps.Series([100, 200, 300, 450, 22])
ssum = np.sum(ss)
print('sum', ssum)

# each_element += 2
ss += 2
print(ss)

# ------------------------------------------------------
# DATA FRAMES

# declaration
s1 = ps.Series({'name': 'Asmaa', 'surname': 'Mirkhan'})
s2 = ps.Series({'name': 'Esma', 'surname': 'Mir'})
s3 = ps.Series({'name': 'Asma', 'surname': 'Mirhan'})
df = ps.DataFrame([s1, s2, s3], index=['student1', 'student2', 'student3'])
print(df)

# querying
# by key
print(df.loc['student1'])

# by index
print(df.iloc[0])

# query one field
print(df.loc['student1', 'name'])
print(df.loc['student1']['name'])

# get a column
print(df.loc[:]['name'])

# delete a row
df = df.drop('student3')
print(df)

# delete a column
del df['surname']
print(df)

# adding new column
df['surname'] = 'Unknown'
print(df)

# -----------------------------------
print('\n\n')

# reading a csv file by pandas
data = ps.read_csv('olympics.csv')
print(data[:5])

# reading with params ('filname', column labels col_num, row labels row_num)
data1 = ps.read_csv('olympics.csv', index_col=0, skiprows=1)
print(data1[:5])


cols = data1.columns  # the output is a series
print(cols)
# renaming columns
for col in data1.columns:
    if col[:2] == '01':
        #data1.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
        data1.rename(columns={col: 'Gold'+col[4:]}, inplace=True)
    elif col[:2] == '02':
        data1.rename(columns={col: 'Silver'+col[4:]}, inplace=True)
    elif col[:2] == '03':
        data1.rename(columns={col: 'Bronze'+col[4:]}, inplace=True)
    elif col[0] == '№':
        data1.rename(columns={col: '$' + col[1:]}, inplace=True)

print(data1.columns)

# -------------------------------------------------
# Boolean Masking
print(data1['Silver'].count())  # 147

# do masking
only_silver = data1.where(data1['Silver'] > 0)
# print as Nan
print(only_silver['Silver'].count())  # 125
# remove Nan
only_silver = only_silver.dropna()
print(only_silver[:10])

# changing index
data1['index_as_col'] = data1.index
print(data1[:5])
data1 = data1.set_index('Total')
print(data1[:5])

# reset index
data1 = data1.reset_index()
print(data1[:5])


######

data = ps.read_csv('census.csv')
print(data.head())

# get distinct values of a column
print(data['DIVISION'].unique())

# Missing values
log = ps.read_csv('log.csv')
print(log[:10])

log = log.set_index('time')
log = log.sort_index()
print(log[:10])


log = log.reset_index()
log = log.set_index('time', 'user')
print(log[:10])

# fill missing values
log = log.fillna(method='ffill')
print(log[:10])
