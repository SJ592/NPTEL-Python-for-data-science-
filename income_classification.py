# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:18:30 2020

@author: ACE
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns

os.chdir("D:\datasets")

data_income=pd.read_csv("income.csv") #importing dataset

data=data_income.copy()
print(data.info()) #to get datatype,count etc

data.isnull().sum()

summary_num=data.describe() #gives descriptive statistics of data (only for numerical var)
print(summary_num)

summary_cat=data.describe(include='O') #for categorical data (top->modal value i.e most freq occuring, freq-> freq of top)
print(summary_cat)

data['gender'].describe(include='O')

data['JobType'].value_counts()
data['occupation'].value_counts()

print(np.unique(data['JobType'])) #to check for unique classes
print(np.unique(data['occupation']))

data=pd.read_csv('income.csv',na_values=[" ?"]) #to read ' ?' as null

#data pre-processing
data.isnull().sum() #to get how many ' ?' are there in JobType and occupation

missing=data[data.isnull().any(axis=1)] #any missing value in a col (axis=1 !!!)
print(missing)

data2=data.dropna(axis=0) #dropping all rows w/ missing values as we don't know relationship b/w features

correlation=data2.corr() #relationship b/w independant var
data2.columns #gives col names

gender=pd.crosstab(index=data2['gender'],columns='count', normalize=True)
print(gender)

gender_salstat=pd.crosstab(index=data2['gender'],columns=data2['SalStat'], normalize='index', margins=True) #given gender salstat is....
print(gender_salstat)

SalStat=sns.countplot(data2['SalStat']) #bar chat 

sns.distplot(data2['age'], bins=10, kde=False, color='red') #histogram

sns.boxplot('SalStat','age',data=data2) #to get relationship b/w salstat and age
data2.groupby('SalStat')['age'].median() #to get median age for salstat categories


sns.countplot(y='JobType',hue='SalStat',data=data2) #group bar plot

pd.crosstab(index=data2['JobType'],columns=data2['SalStat'], normalize='index',margins=True)

sns.countplot(y='EdType',hue='SalStat',data=data2)

pd.crosstab(index=data2['EdType'],columns=data2['SalStat'], normalize='index',margins=True)

sns.countplot(y='occupation',hue='SalStat',data=data2)

pd.crosstab(index=data2['occupation'],columns=data2['SalStat'], normalize='index',margins=True)

data2.columns

sns.distplot(data2['capitalgain'], bins=5, kde=False)

sns.distplot(data2['capitalloss'], bins=5, kde=False)

sns.boxplot('SalStat','hoursperweek',data=data2)



#LOGISTIC REGRESSION

#Reindexing salary status names (target) to 0 or 1 
data2['SalStat']=data2.loc[:,("SalStat")].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

#Dummy variables-> to give categorical vars. values 0 or 1 instead of 1,2,3...

#one hot encoding-> making new cols which are the categories present ( will have 0 or 1 values) in the categorical cols.  

new_data=pd.get_dummies(data2, drop_first=True) 

columns_list=list(new_data.columns) #storing col names
print(columns_list)

features=list(set(columns_list)-set(['SalStat'])) #separating output val from data
print(features)

y=new_data['SalStat'].values #to get the values in the col. salstat
print(y)

x=new_data[features].values
print(x)

from sklearn.model_selection import train_test_split #to partition data

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0) #random_state->same set of samples will be choosen for analysis everytime command is run

from sklearn.linear_model import LogisticRegression 

#make an instance of the model
logistic=LogisticRegression()

#fitting the model
logistic.fit(train_x,train_y) #Fitting a model means that you're making your algorithm learn the relationship between predictors and outcome so that you can predict the future values of the outcome.
logistic.coef_
logistic.intercept_

prediction=logistic.predict(test_x) #will give us SalStat (value we wanted to predict) of the test dataset (test_x contains all cols except SalStat)
print(prediction)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix1=confusion_matrix(test_y,prediction) #actual salstat values ->test_y, predicted salstat values->prediction
print(confusion_matrix1)
#[correct incorrect] -> salary<=50000
#[incorrect correct] -> salary>50000

accuracy_score1=accuracy_score(test_y,prediction)
print(accuracy_score1) #83%

#printing misclassified values from prediction
print("Missclassified samples: %d" % (test_y != prediction).sum())

#LOGISTIC REGRESSION- REMOVING INSIGNIFICANT VARIABLES

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)

columns_list=list(new_data.columns)
print(columns_list)

features=list(set(columns_list)-set(['SalStat']))
print(features)

y=new_data['SalStat'].values
print(y)

x=new_data[features].values
print(x)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

logistic=LogisticRegression()

logistic.fit(train_x,train_y)

prediction=logistic.predict(test_x)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score) #accuracy reduced ?!!



#KNN (K NEarest Neighbors)----------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

#storing the k nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#fitting values for x and y
KNN_classifier.fit(train_x,train_y)

#predicting test values with model
prediction=KNN_classifier.predict(test_x) #prediction var will contain corresponding SalStat values predicted from test dataset

#performance metric check
confusion_matrix=confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)
#[correct incorrect] -> salary<=50000
#[incorrect correct] -> salary>50000

from sklearn.metrics import accuracy_score

accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

#printing misclassified values from prediction
print("Missclassified samples: %d" % (test_y != prediction).sum())


#Effect of K value on classifier
Misclassified_sample=[]

#calc error for K values between 1 to 20
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=pred_i).sum())

print(Misclassified_sample)

KNN_classifier=KNeighborsClassifier(n_neighbors=10) #lowest misclassified valued were w/ 10 neighbors (from the loop above)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_classifier.predict(test_x)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix=confusion_matrix(test_y,prediction)
print("Original values","\n",confusion_matrix)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score) #best accuracy till now = 84%


#KNN w/ 10 neighbors and removing insignificant vars.

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)

columns_list=list(new_data.columns)
print(columns_list)

features=list(set(columns_list)-set(['SalStat']))
print(features)

y=new_data['SalStat'].values
print(y)

x=new_data[features].values
print(x)

train_x1,test_x1,train_y1,test_y1=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors=10) #lowest misclassified valued were w/ 10 neighbors (from the loop above)
KNN_classifier.fit(train_x1,train_y1)
prediction=KNN_classifier.predict(test_x1)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix=confusion_matrix(test_y1,prediction)
print("Original values","\n",confusion_matrix)

accuracy_score=accuracy_score(test_y1,prediction)
print(accuracy_score) #still 84 %
