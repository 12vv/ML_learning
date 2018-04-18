# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# create column lists
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']

# use pandas to get data from web
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'
                   'breast-cancer-wisconsin.data', names=column_names)

data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

# print(data)

# divide data
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                    test_size=0.25, random_state=33)

print(y_train.value_counts())

# linear
# standardize
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
# fit already, don't need to do it again
X_test = ss.transform(X_test)

# initialize
lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train, y_train)
# predict
lr_y_predict = lr.predict(X_test)