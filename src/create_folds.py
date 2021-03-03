# \.conda\envs\myml\python.exe python

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import RANDOM_STATE, NUM_FOLDS, KFOLD_COLUMN, TARGET_COLUMN, config_dict
from time import time
import argparse

if __name__ == "__main__":

    """Saves preprocessed csv file with folds indicated by additional column"""

    pt_ = time()*1000.0
    print("Creating Stratified Folds...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", type = str)
    parser.add_argument("--ofile", type = str)
    args = parser.parse_args()

    df = pd.read_csv(config_dict[str(args.ifile)])
    df[KFOLD_COLUMN] = -1    
    
    skf_ = StratifiedKFold(n_splits = NUM_FOLDS, shuffle = True, random_state = RANDOM_STATE)
        
    for fold_, (train_, val_) in enumerate(skf_.split(X=df, y=df[TARGET_COLUMN].values)):
        df.loc[val_, KFOLD_COLUMN] = fold_
    
    df.to_csv(config_dict[str(args.ofile)], index=False)

    print(f"Folded data created!         Time Taken : {time()*1000.0 - pt_}ms ")