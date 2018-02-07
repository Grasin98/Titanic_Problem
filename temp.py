# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: @Grasin98
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("C:/Users/NISARG/Downloads/Ship/train.csv")
test=pd.read_csv("C:/Users/NISARG/Downloads/Ship/test.csv")


print(train.isnull().sum())
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)

#print(train.head())

train["Age"].fillna(train["Age"].mean(),axis=0,inplace=True)
test["Age"].fillna(test["Age"].mean(),axis=0,inplace=True)

plt.figure(figsize=(8,6))
sns.distplot(train["Survived"],bins=5,color="brown") #Survived People Vs. People Died
plt.show()



#print(train.head(10))

corr=train.corr()
sns.heatmap(corr)
plt.show()

plt.figure(figsize=(6,3))
sns.barplot(x='Sex', y='Survived',data=train)   #Sex Vs Survived
plt.show()

plt.figure(figsize=(6,3))
sns.barplot(x='Pclass', y='Survived',data=train)   #Pclass Vs Survived
plt.show()

plt.figure(figsize=(6,3))
sns.barplot(x='Pclass', y='Sex',data=train)   #Pclass Vs Survived
plt.show()

plt.figure(figsize=(8,6))
sns.distplot(train["Age"],bins=5,color="brown") #Survived People Vs. People Died
plt.show()


train.drop("Ticket",1,inplace=True)
test.drop("Ticket",1,inplace=True)

train["Sex"]=pd.get_dummies(train["Sex"])
test["Sex"]=pd.get_dummies(test["Sex"])

train["Embarked"]=pd.get_dummies(train["Embarked"])
test["Embarked"]=pd.get_dummies(test["Embarked"])

train.drop("Name",1,inplace=True)
test.drop("Name",1,inplace=True)
print(train.head())

train_x=train.drop("Survived",axis=1)
#test_x=train.drop("Survived",axis=1)
train_y=train["Survived"]

test.fillna(method="ffill",axis=0,inplace=True)

print(test.isnull().sum())

print(train_x.shape)
print(train_y.shape)
print(test.shape)

train_x=train_x.values
train_y=train_y.values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.fit_transform(test)
#train_y=sc.fit(train_y)
#print(train_x.shape)
#print(train_y.shape)

from sklearn.svm import SVC
svc=SVC()
svc.fit(train_x,train_y)
y_pred=svc.predict(test)
acc= round(svc.score(train_x,train_y)*100,2)
print(acc)




