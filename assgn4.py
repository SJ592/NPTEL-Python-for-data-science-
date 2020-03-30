# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:29:01 2020

@author: ACE
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns

#setting dimensions for plot
sns.set(rc={'figure.figsize':(11.7,8.27)})

os.chdir("D:\datasets")

Train_Data=pd.read_csv('CrashTest_TrainData.csv')
Test_Data=pd.read_csv('CrashTest_TestData.csv')

Train_Data.info()
Test_Data.info()
Train_Data.describe()
Test_Data.describe()

Train_Data['CarType'].unique() #to get distinct carType

Train_Data.isna().sum()

pd.crosstab(index=Test_Data['CarType'],columns='count', normalize=True) #proportion of carTypes

#last 4

#removing tuples w/ missing values
Train_Data1=Train_Data.dropna(axis=0)
Test_Data1=Test_Data.dropna(axis=0)
Train_Data1.isna().sum()
Test_Data1.isna().sum()

Train_Data1.describe()
Test_Data1.describe()
#Map the categorical variables into integers
Train_Data1['CarType']=Train_Data1.loc[:,("CarType")].map({'SUV':0, 'Hatchback':1})
print(Train_Data1['CarType'])

Test_Data1['CarType']=Test_Data1.loc[:,("CarType")].map({'SUV':0, 'Hatchback':1})
print(Test_Data1['CarType'])

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

#new_data=pd.get_dummies(Train_Data1, drop_first=True)  #no categorical var !!!
#new_test_data=pd.get_dummies(Test_Data1, drop_first=True) 


columns_list=list(Train_Data1.columns) #storing col names
print(columns_list)

features=list(set(columns_list)-set(['CarType'])) #separating output val from data
print(features)

y=Train_Data1['CarType'].values 
print(y)

x=Train_Data1[features].values
print(x)

columns_list1=list(Test_Data1.columns) #storing col names
print(columns_list1)

features1=list(set(columns_list1)-set(['CarType'])) #separating output val from data
print(features1)

y1=Test_Data1['CarType'].values 
print(y1)

x1=Test_Data1[features1].values
print(x1)


#-------------------x,y (train test)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

#storing the k nearest neighbors classifier
model_1=KNeighborsClassifier(n_neighbors=3)

#fitting values for x and y
model_1.fit(train_x,train_y)

#predicting test values with model
prediction=model_1.predict(test_x)

from sklearn.metrics import accuracy_score, confusion_matrix

#performance metric check
confusion_matrix=confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

accuracy_score1=accuracy_score(test_y,prediction)
print(accuracy_score1) #0.625

#printing misclassified values indices from prediction
print(np.where(test_y != prediction))



#storing the k nearest neighbors classifier
model_2=KNeighborsClassifier(n_neighbors=2)

#fitting values for x and y
model_2.fit(train_x,train_y)

#predicting test values with model
prediction=model_2.predict(test_x)

accuracy_score2=accuracy_score(test_y,prediction)
print(accuracy_score2) #0.833

#logistic regression

from sklearn.linear_model import LogisticRegression 

model_3=LogisticRegression()

model_3.fit(train_x,train_y)

prediction=model_3.predict(test_x)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score) #1.0
