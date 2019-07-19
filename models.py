# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:33:27 2019

@author: Byron Dennis
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

class Model(object):

        
    def print_accuracy_score(self, X_train, y_train, X_val, y_val):
        res = [self.clf.score(X_train, y_train), self.clf.score(X_val, y_val)]
        print(res)
    
    def print_auc_score(self, X_train, y_train, X_val, y_val):
        res = [roc_auc_score(y_train, self.clf.predict(X_train)), 
               roc_auc_score(y_val, self.clf.predict(X_val))]
        print(res)


class RFClassifier1(Model):
    
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()        
        self.clf = RandomForestRegressor(n_estimators=25, verbose=True, 
                                         max_depth=10, min_samples_split=2,
                                         min_samples_leaf=5, n_jobs=-1)
        self.clf.fit(X_train, y_train)
        print("Model Accuracy train/valid: ", self.print_accuracy_score(X_train, y_train, X_val, y_val))
        
        #need to support muli-classification problems
        #print("AUC train/valid: ", self.print_auc_score(X_train, y_train, X_val, y_val))