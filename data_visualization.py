# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:00:02 2020

@author: ACE
"""
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

os.chdir('D:\datasets')

cars_data=pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])
cars_data.dropna(axis=0,inplace=True) #axis=0->rows ,inplace=True->changes will be reflected in the dataframe

#Scatterplot

plt.scatter(cars_data['Age'],cars_data['Price'],c='red') #age= X & Price =Y c->color 
plt.title("Scatter plot of price vs age of the car")
plt.xlabel("Age(months)")
plt.ylabel("Price(euros)")
plt.show()

#Histogram ->continuous variables

plt.hist(cars_data['KM']) #default args.
plt.hist(cars_data['KM'],color='green',edgecolor='white',bins=5) #bins=range/intervals
plt.title("Histogram of kilometers")
plt.xlabel("Kilometer")
plt.ylabel("Frequency")
plt.show()

#Barplot ->categorical variables

counts=[979,120,12] #counts=y coordinate
fuelType=("Petrol","Diesel","CNG")
index=np.arange(len(fuelType)) #index=x coordinate

plt.bar(index,counts,color=['red','blue','cyan'])
plt.title("Barplot of fuel types")
plt.xlabel("Fuel Type")
plt.ylabel("Frequency")
plt.xticks(index,fuelType,rotation=90) #to add labels to bars , index=location of labels, fuelType (already defined) = labels
plt.show()