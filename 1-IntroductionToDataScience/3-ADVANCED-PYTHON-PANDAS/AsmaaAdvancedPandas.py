# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK3 Advanced Pandas - Merging Data Frames

import pandas as pd
import numpy as np

# MARGING DATA FRAMES

data = pd.DataFrame([{'name': 'asmaa', 'surname': 'mirkhan', 'city': 'istanbul'},
                     {'name': 'esma', 'surname': 'mirhan', 'city': 'damascus'},
                     {'name': 'Esma', 'surname': 'mir', 'city': 'bursa'}])


# row-based querying
print(data.iloc[0])

# column-based querying
print(data['name'])

# add new column
data['job'] = ['student', 'programmer', 'engineer']

# add new column and fill by specific value
data['gender'] = 'female'
print(data)

# merging data frames
# like joins in SQL

store1_products = pd.DataFrame([{'category': 'Mobiles',
                                 'name': 'Xiaomi mi 5s plus', 'color': 'Pink'},
                                {'category': 'Tablets',
                                 'name': 'Samsung Galaxy Tab', 'color': 'Grey'},
                                {'category': 'Laptops',
                                 'name': 'MSI cx62', 'color': 'Blue'}])
store1_products = store1_products.set_index('name')
store2_products = pd.DataFrame([{'category': 'Electronics',
                                 'name': 'Samsung Galaxy Tab', 'color': 'Black'},
                                {'category': 'Laptops',
                                 'name': 'MSI cx62', 'color': 'Green'},
                                {'category': 'Accessories',
                                 'name': 'Xiaomi airphone', 'color': 'Purple'}])
store2_products = store2_products.set_index('name')

# select * from store1 outer join store2 on name1 = name2
# since it is outer join missing data will be exist in the result
# logic: show products that exist in either store1 OR store2
print(pd.merge(store1_products, store2_products,
               how='outer', left_index=True, right_index=True))

# select * from store1 inner join store2 on name1 = name2
# since it is inner join missing data will not be exist in the result
# logic: show products that exist in BOTH store1 AND store2
print(pd.merge(store1_products, store2_products,
               how='inner', left_index=True, right_index=True))

# select * from store1 left join store2 on name1 = name2
# since it is left join missing data will be exist only in the right of the result
# logic: show products that exist in store1 AND MAYBE store2
print(pd.merge(store1_products, store2_products,
               how='left', left_index=True, right_index=True))


# select * from store1 right join store2 on name1 = name2
# since it is right join missing data will be exist only in the left of the result
# logic: show products that exist in store2 AND MAYBE store1
print(pd.merge(store1_products, store2_products,
               how='right', left_index=True, right_index=True))


# doing merging on a field without changing indices
# go back to default indexing
store1_products = store1_products.reset_index()
store2_products = store2_products.reset_index()
# do merging
print(pd.merge(store1_products, store1_products,
               how='inner', left_on='name', right_on='name'))


# finding average with group by
# target: find average of CENSUS2010POP group by state
# 1st method

# read file
data = pd.read_csv('census.csv')

# filter data
data = data.where(data['SUMLEV'] == 50).dropna()

# loop over unique states
avg = []
for state in data['STNAME'].unique():
    a = np.average(data.where(data['STNAME'] ==
                              state).dropna()['CENSUS2010POP'])
    avg.append(a)
print(avg)


# 2nd method MORE EFFICIENT
avg = []
for group, item in data.groupby('STNAME'):
    a = np.average(data['CENSUS2010POP'])
    avg.append(a)
print(avg)


# aggregations
# 3rd method 'my favorite :p'
avg = data.groupby('STNAME').agg({'CENSUS2010POP': np.average})
print(avg)

# aggregations
grouped_data = data.set_index('STNAME').groupby(
    level=0)['CENSUS2010POP'].agg({'sum': np.sum, 'avg': np.average, 'max': np.max, 'min': np.min})

# aggregations for multiple fields
agg_data = data.set_index('STNAME').groupby(level=0)[
    'POPESTIMATE2011', 'POPESTIMATE2010'].agg({'sum': np.sum, 'avg': np.average})

print(agg_data)
