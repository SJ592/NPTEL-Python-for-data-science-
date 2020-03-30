# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:48:45 2020

@author: ACE
"""


import os
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('D:\datasets')

#Qn.to extract the columns in the following sequence - Time, TotalBill, Tips

Data=pd.read_csv("Tips.csv",index_col=0)
df1=pd.DataFrame(Data, columns= ['Time', 'TotalBill', 'Tips'] )
df2=Data[ ['Time', 'TotalBill', 'Tips'] ]
#df3=Data.iloc[:,0:2]
df4=Data.loc[:, ['Time', 'TotalBill', 'Tips'] ]

#Qn. to merge the two data frames ‘Data’ and ‘Data1’ by columns

Data1=pd.read_excel("Tips1.xlsx")
#Data2 = pd.concat(Data, Data1, join='outer') ->error
Data2 = pd.DataFrame.join(Data, Data1, on=None, how='left')
#Data2_1 = pd.DataFrame.append(Data,Data1)
#Data2_2 = pd.merge(Data, Data1, how='left') ->error

#Qn.total tips received across Day’s 

Data3 = Data2.copy()
#Data3.groupby(['Day', 'Tips']).aggregate(sum)
#Data3.groupby('Day', 'Tips').aggregate(sum)
Data3.groupby('Day')[['Tips']].aggregate(sum)
#Data3.groupby('Day', ['Tips'])['Tips'].aggregate(sum)

#Qn.count of the Time (‘Dinner' or 'Lunch') across gender

Data3.groupby(['Gender', 'Time'])['Time'].count().unstack()
#Data3.groupby('Gender')['Time'].aggregate(sum)
pd.crosstab(index = Data3['Gender'], columns = Data3['Time'], normalize = False)
Data3.pivot_table('Time', index='Gender', columns=Data3.Time.values, aggfunc=len)

plt.hist(Data3['TotalBill'],bins=10,edgecolor='white')

sns.countplot(x="Day",data=Data3)

#Qn.mean of the ‘TotalBill’, ‘Tips’ and ‘Size’ across Days

Data3.groupby('Day').aggregate('mean')
#Data3['Tips'].mean()
Data3.groupby('Day').apply(lambda x: x.mean())
#Data3.groupby('Day').apply(mean)


sns.boxplot(x=Data3['Day'],y=Data3['TotalBill'])


import copy
x=[5,4,3,2,1]
y=[7,8,9]
z=[x,y]
a=copy.deepcopy(z)
b=copy.copy(z)
x[2]=6
print("a=", a, "b=",b)