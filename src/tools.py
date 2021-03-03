# \.conda\envs\myml\python.exe python

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import MEAN_ENCODED_COLUMN, mean_encoding, TARGET_COLUMN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

class CustomMeanEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, target_col = None, column_array=[], encoder = []):
        self.target_col = target_col
        self.mean_encoding = encoder
        self.column_array = column_array
        
    def fit(self, df):
        if self.mean_encoding != []:
            return self
        for i in self.column_array:
            self.mean_encoding.append(df.groupby([i])[self.target_col].mean().to_dict()[TARGET_COLUMN])   
        # for test set pipelining
        mean_encoding = self.mean_encoding
        return self
    
    def transform(self, df):
        for encodings,column in zip(self.mean_encoding,self.column_array):
            df[column+ MEAN_ENCODED_COLUMN] = df[column].apply(lambda value: encodings[value])
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    
    def __init__(self,column_array=[]):
        self.column_array = column_array
        
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        return df.drop(self.column_array, axis = 1)


class CustomPreprocessing(BaseEstimator, TransformerMixin):
    
    # use this for manual feature engineering

    def __init__(self,drop_=True):
        pass

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

class CustomNormalizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, ntype = "sc", column_array = [],*args,**kwargs):
        self.column_array = column_array
        self.ntype = ntype
        if self.ntype == "pt":
            self.cn = PowerTransformer(*args,**kwargs)
        elif self.ntype == "mm":
            self.cn = MinMaxScaler()
        else:
            self.cn = StandardScaler()
    
    def fit(self, df):
        self.cn.fit(df[self.column_array])
        return self
        
    def transform(self, df):
        df[self.column_array] = self.cn.transform(df[self.column_array])
        return df