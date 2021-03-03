# \.conda\envs\myml\python.exe python

import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    """Saves preprocessed csv file with folds indicated by additional column"""

    df = pd.read_csv(CLEAN_DATA)
    df[KFOLD_COLUMN] = -1    
    
    skf_ = StratifiedKFold(n_folds = NUM_FOLDS, shuffle = True, random_state = RANDOM_STATE)
        
    for fold_, (train_, val_) in enumerate(skf_.split(X=df, y=df[TARGET_COLUMN].values)):
        df.loc[val_, KFOLD_COLUMN] = fold_
    
    df.to_csv(FOLDED_DATA, index=False)
