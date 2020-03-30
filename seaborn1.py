# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:34:27 2020

@author: ACE
"""
import os
import pandas as pd #to work with dataframes
import numpy as np  #to do numerical operations
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization

os.chdir('D:\datasets')

cars_data=pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])
cars_data.dropna(axis=0,inplace=True) #removing missing values

#Scatter plot

sns.set(style="darkgrid")
sns.regplot(x=cars_data["Age"],y=cars_data["Price"]) #scatter plot
sns.regplot(x=cars_data["Age"],y=cars_data["Price"],marker="*",fit_reg=False) #to remove regression line 

sns.lmplot(x="Age",y="Price",data=cars_data,fit_reg=False,hue='FuelType',legend=True,palette="Set1") #legend=true => which color represens what data

#Histogram
sns.distplot(cars_data['Age']) #kernel density estimate is default
sns.distplot(cars_data['Age'],kde=False) #to just get normal freq. 
sns.distplot(cars_data['Age'],kde=False,bins=5)

#Barplot

sns.countplot(x="FuelType",data=cars_data)

#Group bar plot

sns.countplot(x="FuelType",data=cars_data, hue="Automatic") #hue-> to specify the 2nd variable (1 var w.r.t 2nd var)

#Box and whiskers plot (for numerical var) 
#whiskers give min and max value excluding outliers

sns.boxplot(y=cars_data['Price'])

#Box and whiskers plot for numerical vs categorical var

sns.boxplot(x=cars_data['FuelType'],y=cars_data['Price'])

#Grouped box and whiskers plot

sns.boxplot(x=cars_data['FuelType'],y=cars_data['Price'],hue=cars_data["Automatic"])

#Box-whiskers and histogram together

f,(ax_box, ax_hist)=plt.subplots(2,gridspec_kw={"height_ratios": (.15,.85)}) #to split the window , default x axis

sns.boxplot(cars_data['Price'],ax=ax_box)  
sns.distplot(cars_data['Price'],kde=False,ax=ax_hist)

#Pairwise plot

sns.pairplot(cars_data,kind="scatter",hue="FuelType") #plots for all variables, scatter plot for numerical var and hue->color classification of fuel types
plt.show()