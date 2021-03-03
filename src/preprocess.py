# \.conda\envs\myml\python.exe python

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

global encoding_g

class CustomMeanEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, target_col = None, column_array=[], encoder = []):
        self.target_col = target_col
        self.mean_encodings = encoder
        self.column_array = column_array
        
    def fit(self, df):
        if self.mean_encodings != []:
            return self
        for i in self.column_array:
            self.mean_encodings.append(df.groupby([i])[self.target_col].mean().to_dict())   
        # for test set pipelining
        encoding_g = self.mean_encodings
        return self
    
    def transform(self, df):
        for encodings,column in zip(self.mean_encodings,self.column_array):
            df[column+'_mean_enc'] = df[column].apply(lambda value: encodings[value])
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    
    def __init__(self,column_array=[]):
        self.column_array = column_array
        
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        return df.drop(self.column_array, axis = 1)


class CustomPreprocessing(BaseEstimator, TransformerMixin):
    
    def __init__(self,drop_=True):
        self.drop_ = drop_

    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        return self

    
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self,column_array=[], column_array_drop_first=[]):
        self.column_array = column_array
        self.column_array_drop_first = column_array_drop_first
    
    def fit(self, df):
        return self
    
    def transform(self,df):
        df = pd.get_dummies(df, columns = self.column_array)
        if self.column_array_drop_first != [] :
            df = pd.get_dummies(df, columns = self.column_array_drop_first, drop_first = True)
        return df

