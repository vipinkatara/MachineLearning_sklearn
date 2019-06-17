# Decision Tree with Gini IndexPython
# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                                max_depth=3, min_samples_leaf=5)
# clf_gini.fit(X_train, y_train)


import math

import numpy as np

import pandas as pd

#from pandas import DataFrame

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from numpy import loadtxt, where

from pylab import scatter, show, legend, xlabel, ylabel

# scale larger positive and values to between -1,1 depending on the largest
# value in the data

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv("train.csv", header=0)

# clean up data

df.columns = ['PassengerId',	'Survived',	'Name',	'Sex',	'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked',	'Pclass']
df = df[["Sex", "Age", "Survived", "Pclass","Fare"]]
cd = df['Pclass']
df.dropna(inplace=True)
print(cd.head)
x = df["Survived"]

X = df[["Pclass", "Sex", "Age","Fare"]]

X = np.array(X)

X = min_max_scaler.fit_transform(X)

Y = df["Survived"]

Y = np.array(Y)
X = X.astype(float)
Y = Y.astype(float)

X_train, X1_test, Y_train, Y1_test = train_test_split(X, Y,test_size=0.0001)

# train scikit learn model
df = pd.read_csv("test.csv", header=0)

# clean up data

df.columns = ["Sex","Age","Survived","Pclass", "Cabin","PassengerId","Name", "SibSp", "Parch", "Ticket", "Fare", "Embarked"]
df.dropna(inplace=True)
x = df["Survived"]

X1 = df[["Pclass","Sex","Age","Fare"]]

X1 = np.array(X)

X1 = min_max_scaler.fit_transform(X)
# X = X.astype(float)
# Y = Y.astype(float)
Y1 = df["Survived"]

Y1 = np.array(Y)
Y1 = Y1.astype(float)
X1 = X1.astype(float)
X1_train, X_test, Y1_train, Y_test = train_test_split(X1, Y1, test_size=0.99, random_state=1010)



#clf = GaussianNB()
clf = SVC(probability=False,  # cache_size=200,
              kernel="rbf", C=2.8, gamma=.0073)
clf.fit(X_train, Y_train)


accuracy = clf.score(X_test,Y_test)

print('accuracy = ', accuracy)