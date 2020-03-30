# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:13:48 2020

@author: ACE
"""

import os
import pandas as pd
import numpy as np

os.chdir('D:\datasets')

cars_data=pd.read_csv('Toyota.csv') 

cars_data=pd.read_csv('Toyota.csv', index_col=0) #to make first col as index col

cars_data1=cars_data.copy(deep=True)
cars_data1.index
cars_data1.columns #to get col lables
cars_data1.size #total no. of elements
cars_data1.shape #no. of rows and cols
cars_data1.memory_usage()
cars_data1.ndim #no. of dimensions 
cars_data1.head(10)
cars_data1.tail(5)
cars_data1.at[4,'FuelType'] #gets scalar value which corresponds to 5th row and FuelType column
cars_data1.iat[5,6] #6th row 7th col
cars_data1.loc[:,'FuelType'] #to access multiple records (here of attri FuelType)

cars_data1.dtypes
cars_data1.get_dtype_counts() #to get count of datatypes in dataset
cars_data1.select_dtypes(exclude=[object]) #to return subset of dataframe of selected datatypes
cars_data1.info() #gives summary of dataframe

print(np.unique(cars_data1['KM'])) #used to get unique elements of column
print(np.unique(cars_data1['HP'])) 
print(np.unique(cars_data1['MetColor'])) 
print(np.unique(cars_data1['Automatic'])) 
print(np.unique(cars_data1['Doors'])) 


cars_data1=pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"]) #making ?? & ???? records null
cars_data1.info()

cars_data1['MetColor']=cars_data1['MetColor'].astype('object') #to convert datatype
cars_data1['Automatic']=cars_data1['Automatic'].astype('object')

cars_data1['FuelType'].nbytes #total bytes consumed by elements of col
cars_data1['FuelType'].astype('category').nbytes
cars_data1.info()

cars_data1['Doors'].replace('three',3,inplace=True) #to replace values , inplace=True -> it is reflected in the dataset
cars_data1['Doors'].replace('four',4,inplace=True)
cars_data1['Doors'].replace('five',5,inplace=True) #or use numpy.where()

cars_data1['Doors']=cars_data1['Doors'].astype('int64') #as python can't differentiate b/w new values and existing values

cars_data1.isnull().sum() #to count no. of missing values in each cols

cars_data1.insert(10,"Price_class","") #to add new col price_class, "" -> initial value, 10->col at 10th position
cars_data1.head()

for i in range(0,len(cars_data['Price']),1): #1=iteration step =>no skip, basically ; creating bins for price col 
    if cars_data1['Price'][i]<=8450:
        cars_data1['Price_class'][i]="Low"
    elif cars_data1['Price'][i]>11950:
        cars_data1['Price_class'][i]="High"
    else:
        cars_data1['Price_class'][i]="Medium"
        
cars_data1.tail()

'''i=0
while i<len(cars_data['Price']):
    if cars_data1['Price'][i]<=8450:
        cars_data1['Price_class'][i]="Low"
    elif cars_data1['Price'][i]>11950:
        cars_data1['Price_class'][i]="High"
    else:
        cars_data1['Price_class'][i]="Medium"
    i=i+1'''
    
cars_data1['Price_class'].value_counts() #to count unique values


cars_data1.insert(11,"Age_converted",0)

def c_convert(val): #to convert age to years
    val_converted=val/12
    return val_converted

cars_data1['Age_converted']=c_convert(cars_data1['Age'])
cars_data1['Age_converted']=round(cars_data1['Age_converted'],1) #rounding value off till 1 decimal
cars_data1.head(10)

cars_data1.insert(12,"Km_per_month",0)

def c_convert1(val1,val2): 
    val_converted=val1/12
    ratio=val2/val1 #km per month ratio
    return [val_converted,ratio] #returns list o/p (for multiple return values)

cars_data1['Age_converted'],cars_data1['Km_per_month']=c_convert1(cars_data1['Age'],cars_data1['KM'])
cars_data1['Age_converted']=round(cars_data1['Age_converted'],1)
cars_data1.head()
