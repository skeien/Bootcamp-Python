# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:25:30 2020

@author: sebas
"""

import pandas as pd # importing pandas as pd
import numpy  as np # importing numpy as np

nums = [1, 2, 3, 4,5]
s = pd.Series(nums)
print(s)

# Creating Pandas Series with custom index
nums = [1, 2, 3, 4, 5]
s = pd.Series(nums, index=[1, 2, 3, 4, 5])
print(s)

fruits = ['Orange','Banana','Mangao']
fruits = pd.Series(fruits, index=[1, 2, 3])
print(fruits)

dct = {'name':'Asabeneh','country':'Finland','city':'Helsinki'}
s = pd.Series(dct)
print(s)

s = pd.Series(10, index = [1, 2,3])
print(s)

s = pd.Series(np.linspace(5, 20, 10)) # linspace(starting, end, items)
print(s)

# DataFrames
# Pandas data frames can be created in different ways.

# Creating DataFrames from List of Lists
data = [
    ['Asabeneh', 'Finland', 'Helsink'], 
    ['David', 'UK', 'London'],
    ['John', 'Sweden', 'Stockholm']
]
df = pd.DataFrame(data, columns=['Names','Country','City'])
print(df)

# Creating DataFrame Using Dictionary
data = {'Name': ['Asabeneh', 'David', 'John'], 'Country':[
    'Finland', 'UK', 'Sweden'], 'City': ['Helsiki', 'London', 'Stockholm']}
df = pd.DataFrame(data)
print(df)

# Creating DataFrames from a List of Dictionaries
data = [
    {'Name': 'Asabeneh', 'Country': 'Finland', 'City': 'Helsinki'},
    {'Name': 'David', 'Country': 'UK', 'City': 'London'},
    {'Name': 'John', 'Country': 'Sweden', 'City': 'Stockholm'}]
df = pd.DataFrame(data)
print(df)

# Reading CSV File Using Pandas
# To download the csv file, needed in this example, console/command line is enough:

# curl -O https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/weight-height.csv
import pandas as pd

df = pd.read_csv('weight-height.csv')
print(df)

# Data Exploration
# Let's read only the first 5 rows using head()

print(df.head()) # give five rows we can increase the number of rows by passing argument to the head() method

# As you can see the csv file has three rows: Gender, Height and Weight. But we don't know the number of rows. Let's use shape meathod.

print(df.shape) # as you can see 10000 rows and three columns

# Let's get all the columns using columns.

print(df.columns)

# Let's read only the last 5 rows using tail()

print(df.tail()) # tails give the last five rows, we can increase the rows by passing argument to tail method
# Now, lets get a specific column using the column key

heights = df['Height'] # this is now a series
print(heights)

weights = df['Weight'] # this is now a series
print(weights)

print(len(heights) == len(weights))

print(heights.describe()) # give statisical information about height data

print(weights.describe())

print(df.describe())  # describe can also give statistical information from a dataFrame

# Creating a DataFrame
# As always, first we import the necessary packages. Now, lets import pandas and numpy, two best friends ever.

import pandas as pd
import numpy as np
data = [
    {"Name": "Sebastian", "Country":"Argentina","City":"Buenos Aires"},
    {"Name": "David", "Country":"UK","City":"London"},
    {"Name": "John", "Country":"Sweden","City":"Stockholm"}]
df = pd.DataFrame(data)
print(df)

# Adding a New Column
# Let's add a weight column in the DataFrame

weights = [80, 78, 69]
df['Weight'] = weights

# Let's add a height column into the DataFrame aswell

heights = [185, 175, 169]
df['Height'] = heights
print(df)

# Modifying column values
df['Height'] = df['Height'] * 0.01
print(df)

# Using functions makes our code clean, but you can calculate the bmi without one
def calculate_bmi ():
    weights = df['Weight']
    heights = df['Height']
    bmi = []
    for w,h in zip(weights, heights):
        b = w/(h*h)
        bmi.append(b)
    return bmi
    
bmi = calculate_bmi()
df['BMI'] = bmi
print(df)

# The BMI column values of the DataFrame are float with many significant digits after decimal. Let's change it to one significant digit after point.

df['BMI'] = round(df['BMI'], 2)
print(df)

birth_year = ['1789', '1985', '1990']
current_year = pd.Series(2020, index=[0, 1,2])
df['Birth Year'] = birth_year
df['Current Year'] = current_year
print(df)

print(df.Weight.dtype)

df['Birth Year'].dtype # it gives string object , we should change this to number

df['Birth Year'] = df['Birth Year'].astype('int')
print(df['Birth Year'].dtype) # let's check the data type now

# Now same for the current year:

df['Current Year'] = df['Current Year'].astype('int')
print(df['Current Year'].dtype)

# Now, the column values of birth year and current year are integers. We can calculate the age.

ages = df['Current Year'] - df['Birth Year']
print(ages)

df['Ages'] = ages
print(df)

mean = (35 + 30)/ 2
print('Mean: ',mean)	#it is good to add some description to the output, so we know what is what
#age

print(df[df['Ages'] > 120])

print(df[df['Ages'] < 120])


#exercise 1
import pandas as pd

hn = pd.read_csv('./files/data/hacker_news.csv')
print(hn)
#exercise 2
print(hn.head())
#exercise 
print(hn.tail())
#exercise 4

columnsNamesArr = pd.Series(hn.columns.values)
print(columnsNamesArr)
#exercise 5

print(hn.shape)
#titles python
print(hn[hn['title'].str.contains("python" or "Python")])
#titles python
print(hn[hn['title'].str.contains("Javascript" or "JavasScript" or "javasscript")])

print(hn.sort_values(by=['num_comments']))
print(hn.sort_values(by=['num_points']))