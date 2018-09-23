# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 01:39:47 2018

@author: mudit
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
path = "C:/Users/mudit/Desktop/Data Analytics/Sugery_Data/"
dataset = pd.read_csv(path + 'SurgeryData.csv', header = None, index_col = None)
col = ['Patient','Device','Duration','Gender']
dataset = dataset.astype(float)
X = dataset[col]
dataset['Surgery'] = dataset['Surgery'].astype(int)
y = dataset['Surgery']

dataset.columns = ["Patient", "Device", "Duration", "Surgery", "Gender"]
dataset = dataset.drop(dataset.index[0])

#Device plot
fig, ax = plt.subplots()
X['Device'].value_counts().plot(ax=ax, kind='bar')

#Duration plot 
fig, ax = plt.subplots()
X['Duration'].value_counts().plot(ax=ax, kind='hist')


#Gender plot
fig, ax = plt.subplots()
X['Gender'].value_counts().plot(ax=ax, kind='bar')

#Surgery plot
fig, ax = plt.subplots()
y.value_counts().plot(ax=ax, kind='bar')

#Patient plot
ct = X["Patient"] 
plt.boxplot(ct.values)
plt.show()

#Correlation with the dependent variable
X['Device'].corr(y)
X['Patient'].corr(y)
X['Duration'].corr(y)
X['Gender'].corr(y)

#full dataframe correlation
plt.matshow(dataset.corr())
dataset.corr()

#split dataset into Train & Test
X_train = X.sample(frac = 0.80, random_state=2)
X_train1 = X_train[["Patient", "Device", "Duration", "Gender"]]
X_test = X.loc[~X.index.isin(X_train.index), :]
X_test1 = X_test[["Patient", "Device", "Duration", "Gender"]]
y_train = y[y.index.isin(X_train.index)]
y_test = y[~y.index.isin(y_train.index)]

# Fitting Logistic Regression
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#matplotlib inline
pd.crosstab(X.Device, y).plot(kind='bar')
plt.title('Frequency for Device per Surgery')
plt.xlabel('Device')
plt.ylabel('Surgery')
plt.savefig('Device_per_Surgery')


#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(X_train1, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#check P-value
import statsmodels.api as sm
a =dataset[col]
b =dataset['Surgery']
logit_model=sm.Logit(b, a)
result=logit_model.fit()
print(result.summary2())

cols = ["Patient", "Device"]
import statsmodels.api as sm
a =dataset[cols]
b =dataset['Surgery']
logit_model=sm.Logit(b, a)
result=logit_model.fit()
print(result.summary2())


X_train2 = X_train1[cols]
X_test2 = X_test1[cols]
y_train
y_test

#logit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train2, y_train)

#y_predict
y_pred = logreg.predict(X_test2)
print('Accuracy of logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test2, y_test)))

#build confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#number of occurences of each class
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test2))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test2)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

