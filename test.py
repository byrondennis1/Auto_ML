# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:41:53 2019

@author: Byron Dennis
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import models


path = 'H:/iris-flower-dataset/'
df = pd.read_csv(path + 'IRIS.csv')

#set target
y = df['species']
X = df.drop(['species'], axis=1)


#encode target
le = LabelEncoder()
y = le.fit_transform(y)

#split train/test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


#create models
models.RFClassifier1(X_train, y_train, X_val, y_val)
