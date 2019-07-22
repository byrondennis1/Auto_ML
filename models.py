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
import datetime


class Model(object):

        
    def print_accuracy_score(self, X_train, y_train, X_val, y_val):
        res = [round(self.clf.score(X_train, y_train), 4), round(self.clf.score(X_val, y_val), 4)]
        return(res)


class RFClassifier1(Model):
    
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()        
        self.clf = RandomForestRegressor(n_estimators=25, max_depth=10, min_samples_split=2,
                                         min_samples_leaf=5, n_jobs=-1)
        
        print("Model: Random Forest")
        print ("Parameters: n_estimators=25, max_depth=10, min_samples_split=2, min_samples_leaf=5, n_jobs=-1")
        print("training start", datetime.datetime.now())
        self.clf.fit(X_train, y_train)
        print("training end", datetime.datetime.now())
        print("Model Accuracy train/valid: ", self.print_accuracy_score(X_train, y_train, X_val, y_val))