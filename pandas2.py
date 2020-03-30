# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:19:33 2020

@author: ACE
"""


import os
import pandas as pd
import numpy as np

os.chdir('D:\datasets')

cars_data=pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

cars_data2=cars_data.copy()

pd.crosstab(index=cars_data2['FuelType'],columns='count',dropna=True) #to create frequency table (to get count of types under certain catagorical var.)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['FuelType'],dropna=True) #freq. distribution of gearbox types (automatic col) w.r.t fueltype

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['FuelType'],normalize=True,dropna=True) # for joint probability ; normalize-> converting nos. to proportion (probability)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['FuelType'],normalize=True,dropna=True,margins=True) #margins->row sum and col sum (marginal probability)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['FuelType'],normalize='index',dropna=True,margins=True) #conditional probability; normalize=index-> given automatic car the fuel type is...

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['FuelType'],normalize='columns',dropna=True,margins=True) #conditional probability; normalize=columns -> just the opposite of above 

numerical_data=cars_data2.select_dtypes(exclude=[object]) 

print(numerical_data.shape)

corr_matrix=numerical_data.corr() #correlation b/w numerical vars. (-ve corr=> inversely proportional,+ve corr=> directly proportional)
