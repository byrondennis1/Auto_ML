#!/usr/bin/env python
# coding: utf-8

# # ModelViz Library

# This library can be used to quickly create visualization that are useful for understanding model performance.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot
import eli5
from eli5.sklearn import PermutationImportance


# In[ ]:


class ModelViz():
    #class used to simplify and speed up visualization of model diagnostics
    def __init__(self, model, X_train, y_train, X_val=None, , y_val=None):
        self.model, self.X_train, self.y_train self.X_val, self.y_val, self.X_valid, self.y_valid = model, X_train, y_train, X_val, y_val
        
    
    def feature_importance_permutation(self, random_state=123):
        '''
        calculate and display feature importance using the permutation method
        must have eli5 installed
        '''
        perm = PermutationImportance(self.model, random_state).fit(X_val, y_val)
        eli5.show_weights(perm, feature_names = X_val.columns.tolist())
        
    

