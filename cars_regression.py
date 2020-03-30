# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:55:17 2020

@author: ACE
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns

#setting dimensions for plot
sns.set(rc={'figure.figsize':(11.7,8.27)})

os.chdir("D:\datasets")
cars_data=pd.read_csv('cars_sampled.csv')

cars=cars_data.copy()

cars.info() # gives structure of data

cars.describe() #summarizes the data
pd.set_option('display.float_format', lambda x:'%.3f' %x) #converts all values to upto 3 decimal places
cars.describe()

pd.set_option('display.max_columns', 500)   #to display all the columns instead of '...' in o/p
cars.describe()

#dropping unwanted variables
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1) #axis=1 =>columns !!!

#removing duplicate records
cars.drop_duplicates(keep='first', inplace=True)

#DATA CLEANING

cars.isnull().sum()

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index() #sort_index()->to sort acc. to year and not year count (which value_count() gives as o/p)
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x="yearOfRegistration",y="price", scatter=True, fit_reg=False,data=cars) #scatterplot

#working rage=1950 -2018 (year of reg.)

price_count=cars['price'].value_counts().sort_index() #index-> actual price values , price-> freq (in the tbl in price_count)
sns.distplot(cars['price']) #histogram
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)

#working rage=100 -150000 (price)

power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS']) #histogram
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x="powerPS",y="price", scatter=True, fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#working rage=10 - 500 (powerPS)

#working range of data
cars=cars[
          (cars.yearOfRegistration <= 2018)
        & (cars.yearOfRegistration >= 1950)
        & (cars.price <= 150000)
        & (cars.price >= 100)
        & (cars.powerPS <= 500)
        & (cars.powerPS >= 10)]

#Combining year of registration and month of registration (for variable reduction)
cars['monthOfRegistration']/=12

#creating new var - age
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#Price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#PowerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

sns.regplot(x='Age',y="price",data=cars,scatter=True,fit_reg=False) #scatter plot

sns.regplot(x='powerPS',y="price",data=cars,scatter=True,fit_reg=False) #scatter plot

#check if any type under categorical var is insignificant compared to other

#seller
cars['seller'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['seller'],columns='count', normalize=True)
sns.countplot(x='seller',data=cars) #bar plot

#offerType
cars['offerType'].value_counts() 
sns.countplot(x='offerType',data=cars)

#abtest
cars['abtest'].value_counts() 
sns.countplot(x='abtest',data=cars)

sns.boxplot(x=cars['abtest'],y=cars['price'])

#vehicalType
cars['vehicleType'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['vehicleType'],columns='count', normalize=True) #proportions (%)
sns.countplot(x='vehicleType',data=cars) #bar plot
sns.boxplot(x=cars['vehicleType'],y=cars['price'])

#gearbox
cars['gearbox'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['gearbox'],columns='count', normalize=True) #proportions (%)
sns.countplot(x='gearbox',data=cars) #bar plot
sns.boxplot(x=cars['gearbox'],y=cars['price'])

#model
cars['model'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['model'],columns='count', normalize=True) #proportions (%),normalize->div each val by total
sns.countplot(x='model',data=cars) #bar plot
sns.boxplot(x=cars['model'],y=cars['price'])

#kilometer (how may km the car has travelled)
cars['kilometer'].value_counts().sort_index() #sort_index -> ascending order 
pd.crosstab(cars['kilometer'],columns='count', normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y="price",data=cars,scatter=True,fit_reg=False) 

#fuelType
cars['fuelType'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['fuelType'],columns='count', normalize=True) #proportions (%),normalize->div each val by total
sns.countplot(x='fuelType',data=cars) #bar plot
sns.boxplot(x=cars['fuelType'],y=cars['price'])

#brand
cars['brand'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['brand'],columns='count', normalize=True) #proportions (%),normalize->div each val by total
sns.countplot(y='brand',data=cars) #bar plot
sns.boxplot(y=cars['brand'],x=cars['price'])

#notRepairedDamage
#(yes-> car is damaged, no-> car was damaged but now fixed)
cars['notRepairedDamage'].value_counts() #count of all types under this categorical var
pd.crosstab(cars['notRepairedDamage'],columns='count', normalize=True) #proportions (%),normalize->div each val by total
sns.countplot(x='notRepairedDamage',data=cars) #bar plot
sns.boxplot(x=cars['notRepairedDamage'],y=cars['price'])

#removing insignificant variables
cols=['seller','offerType','abtest']
cars=cars.drop(columns=cols,axis=1)
cars_copy=cars.copy()

#correlation
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:] #correlation of rest of the var w/ price


#omitting missing values

cars_omit=cars.dropna(axis=0)

#converting categorical var to dummy var (performing one hot encoding)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#MODEL BUILDING WITH OMITTED DATA

x1=cars_omit.drop(['price'],axis='columns',inplace=False) #inplace =False -> price col will not be dropped from cars_omit 
y1=cars_omit['price']

prices=pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()

np.isinf(y1).sum() #checking for infinite values

y1=np.log(y1) #transforming price as logarithmic value

X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape ,X_test.shape ,y_train.shape, y_test.shape)

#base line model for omitted data

#finding mean of test data
base_pred=np.mean(y_test)
print(base_pred)

#repeating same value for entire length of test data
base_pred=np.repeat(base_pred, len(y_test))

#finding RMSE (b/w predicted values and mean of test values)
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))

print(base_root_mean_square_error)

#LINEAR REGRESSION w/ omitted data

lgr=LinearRegression(fit_intercept=True)

#model
model_lin1=lgr.fit(X_train,y_train)

#predition
cars_predictions_lin1=lgr.predict(X_test)

#computing MSE and RMSE
lin_mse1=mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R squared value ( the larger the R2, the better the regression model fits your observations.)
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagnostics-residual plot analysis
residuals1=y_test-cars_predictions_lin1 #residuals -> errors value should be less
sns.regplot(x=cars_predictions_lin1,y=residuals1, scatter=True, fit_reg=False, data=cars)
residuals1.describe()

#RANDOM FOREST w/ omitted data
rf=RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)

#model
model_rf1=rf.fit(X_train,y_train)

#prediction
cars_prediction_rf1=rf.predict(X_test)

#coputing MSE and RMSE
rf_mse1=mean_squared_error(y_test, cars_prediction_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#R squared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#MODEL BUILDING WITH IMPUTED DATA (where null is replaced w/ smthn)
cars_imputed=cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' 
                        else x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

#converting categorical columns to dummy variables
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)

#MODEL BUILDING WITH IMPUTED DATA
x2=cars_imputed.drop(['price'],axis='columns',inplace=False) #inplace =False -> price col will not be dropped from cars_omit 
y2=cars_imputed['price']

#plotting the variable price
prices=pd.DataFrame({"1. Before":y2, "2. After":np.log(y2)})
prices.hist()

y2=np.log(y2) #transforming price as logarithmic value

X_train1,X_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train1.shape ,X_test1.shape ,y_train1.shape, y_test1.shape)

#base line model for omitted data

#finding mean of test data
base_pred=np.mean(y_test1)
print(base_pred)

#repeating same value for entire length of test data
base_pred=np.repeat(base_pred, len(y_test1))

#finding RMSE (b/w predicted values and mean of test values)
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1,base_pred))

print(base_root_mean_square_error_imputed)

#LINEAR REGRESSION w/ imputed data

lgr2=LinearRegression(fit_intercept=True)

#model
model_lin2=lgr2.fit(X_train1,y_train1)

#predition
cars_predictions_lin2=lgr2.predict(X_test1)

#computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1, cars_predictions_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

#R squared value (how much inconsistencies model is able to capture )
r2_lin_test2=model_lin2.score(X_test1,y_test1)
r2_lin_train2=model_lin2.score(X_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

#RANDOM FOREST w/ imputed data
rf2=RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)

#model
model_rf2=rf2.fit(X_train1,y_train1)

#prediction
cars_prediction_rf2=rf2.predict(X_test1)

#computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1, cars_prediction_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#R squared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)

