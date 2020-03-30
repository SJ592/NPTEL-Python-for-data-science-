# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:15:38 2020

@author: ACE
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('D:\datasets')

churn_data=pd.read_csv('churn.csv',index_col=0)
churn_data.info()
churn_data['TotalCharges'].isnull().sum()

diamond_data=pd.read_csv('diamond.csv',index_col=0)
sns.boxplot(x=diamond_data['price'],y=diamond_data['cut'])
diamond_data.loc[:,['carat','cut','color']]
diamond_data[['carat','cut','color']]
diamond_data.iloc[:,0:2]

mtcars_data=pd.read_csv('mtcars.csv')
plt.scatter(mtcars_data['mpg'],mtcars_data['wt'],c='red')
plt.title("Scatter plot for Miles per gallon vs Weight")
plt.xlabel("mpg")
plt.ylabel("wt")

plt.plot([1,2,3,4],[10,20,30,40])

