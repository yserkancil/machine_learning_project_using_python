# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#load value
datas = pd.read_csv('test_full.csv')

############# DATA PREPROCESSİNG ###############
column_team = datas['team'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=column_team.index, y=column_team.values)
plt.title('Tahmin yapılan takım')
plt.xlabel('Ülke')
plt.ylabel('Sayı')
plt.xticks(rotation=45, ha='right')  # X-ekseni etiketlerini döndürme
plt.show()

country_opp = datas['opp'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=country_opp.index, y=country_opp.values)
plt.title('Rakip Takım')
plt.xlabel('Ülke')
plt.ylabel('Sayı')
plt.xticks(rotation=45, ha='right')  # X-ekseni etiketlerini döndürme
plt.show()

country_hosts = datas['host'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=country_hosts.index, y=country_hosts.values)
plt.title('Ev Sahibi Ülke')
plt.xlabel('Ülke')
plt.ylabel('Sayı')
plt.xticks(rotation=45, ha='right')  # X-ekseni etiketlerini döndürme
plt.show()

column_month = datas['month'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=column_month.index, y=column_month.values)
plt.title('aylar')
plt.xlabel('Ülke')
plt.ylabel('Sayı')
plt.xticks(rotation=45, ha='right')  # X-ekseni etiketlerini döndürme
plt.show()

datas['score'].fillna(datas['score'].mean(), inplace=True)

month_order = {month: i for i, month in enumerate(sorted(datas['month'].unique()))}
datas['month'] = datas['month'].map(month_order)

#i changed values where in column who named team and opp.Because more healty like that for our algorithms.
team_mapping = {team: i+1 for i, team in enumerate(datas['team'].unique())}
datas['team'] = datas['team'].map(team_mapping)
datas['opp'] = datas['opp'].map(team_mapping)

#i delete this columns ,cuz i dont need
datas = datas.drop(['unnamed', 'host',], axis=1)



###############   DATA SET TRAİNİNG AND İMPLEMENTATİON      ##############

#divide test and train 
from sklearn.model_selection import train_test_split

independent_variables = datas.drop(['result'], axis=1)
dependent_variables = datas['result']

independent_variables_train,independent_variables_test,dependent_variables_train,dependent_variables_test= train_test_split(independent_variables,dependent_variables,test_size=0.33 ,random_state=(0))


#scaling of datas and fit with logistic regression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()   

independent_variables_train = sc.fit_transform(independent_variables_train)
independent_variables_test = sc.transform(independent_variables_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(independent_variables_train,dependent_variables_train)

dependent_variables_pred = logr.predict(independent_variables_test)


from sklearn.metrics import confusion_matrix

#show with confusing matrix false and true values
cm = confusion_matrix(dependent_variables_test, dependent_variables_pred)
#print(cm)

#calculate Precision(eküri) 
micro_precision = np.sum(np.diag(cm)) / np.sum(cm)

print("Microaveraged Precision:", micro_precision)

#KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1 , metric='minkowski')
knn.fit(independent_variables_train , dependent_variables_train)
dependent_variables_pred = knn.predict(independent_variables_test)

cm = confusion_matrix(dependent_variables_test, dependent_variables_pred)
print(micro_precision)


#increase succes with support vector machine
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(independent_variables_train , dependent_variables_train)
dependent_variables_pred = svc.predict(independent_variables_test)

#print("after from SVM")
cm = confusion_matrix(dependent_variables_test, dependent_variables_pred)
print(cm,micro_precision)


#Gausian Naive Bayes
from sklearn.naive_bayes import GaussianNB 

gnb=GaussianNB()
gnb.fit(independent_variables_train,dependent_variables_train)

dependent_pred = gnb.predict(independent_variables_test)

cm = confusion_matrix(dependent_variables_test, dependent_variables_pred)

print("after from GNB")
print(cm,micro_precision)


#Decison tree algorithm
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(independent_variables_train,dependent_variables_train)

dependent_pred = dtc.predict(independent_variables_test)

cm = confusion_matrix(dependent_variables_test, dependent_variables_pred)

print("after from DTC and last status")
print(cm,micro_precision)












