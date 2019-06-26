# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK4 Statistical Analysis - Distributions and Hypothesis Testing

import pandas as pd
import numpy as np

# simulate probability problems
# binomial variable: one trial p(1)=p(0)=0.5
bino = np.random.binomial(1, 0.5)

# sum of 1000 trials
bino = np.random.binomial(1000, 0.5)
print(bino)

# average of 1000 trials
bino = np.random.binomial(1000, 0.5)/1000
print(bino)

# normal distibution
# (expected_value, trial)
norm_dis = np.random.normal(0.75, size=100)
print(norm_dis)

# caculating standard deviation
std = np.sqrt(sum((norm_dis-np.mean(norm_dis))**2)/len(norm_dis))
print(std)

# built-in std function
std=np.std(norm_dis)
print(std)

# --------------------------------------

# we have grades of student with submission date

grades = pd.read_csv('grades.csv')
print(grades[:15])

# we suppose that who subnits earlier is will get better grade
# we specify our date treshold as 2015-12-31

early = grades.where(grades['assignment1_submission'] <= '2015-12-31').dropna() #[df['assignment1_submission'] <= '2015-12-31']
late = grades.where(grades['assignment1_submission'] > '2015-12-31').dropna()

# we calculate means check truth of our hypothesis
print(early.mean(), late.mean())
