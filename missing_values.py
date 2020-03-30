# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 07:35:16 2020

@author: ACE
"""


import os
import pandas as pd

os.chdir('D:\datasets')

cars_data=pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

cars_data2=cars_data.copy()
cars_data3=cars_data2.copy()

cars_data2.isnull().sum() #or isna()
missing= cars_data2[cars_data2.isnull().any(axis=1)] #axis=1->cols , subsetting rows that have any missing values(in cols)
cars_data2.describe()

#if mean is not to far away from median us mean to fill null values else use median  (mean is far away => outliers are affecting it)

#numerical variables- w/ mean or median
cars_data2['Age'].mean()
cars_data2['Age'].fillna(cars_data2['Age'].mean(),inplace=True)

cars_data2['KM'].median()
cars_data2['KM'].fillna(cars_data2['KM'].median(),inplace=True)

cars_data2['HP'].mean()
cars_data2['HP'].median()
cars_data2['HP'].fillna(cars_data2['HP'].mean(),inplace=True)

cars_data2.isna().sum()

#categorical variables -w/ most frequently occuring class
cars_data2['FuelType'].value_counts() #to get freq. of categories
cars_data2['FuelType'].value_counts().index[0] #first value in o/p of value_counts() (which will be most freq occuring type !)
cars_data2['FuelType'].fillna(cars_data2['FuelType'].value_counts().index[0],inplace=True)

cars_data2['MetColor'].mode() #mode->most freq occuring
cars_data2['MetColor'].fillna(cars_data2['MetColor'].mode()[0],inplace=True)

cars_data2.isnull().sum()

#to fill null values of numerical and categorical var in one go 
cars_data3=cars_data3.apply(lambda x:x.fillna(x.mean()) if x.dtype=='float' else x.fillna(x.value_counts().index[0])) 
cars_data3.isna().sum()
