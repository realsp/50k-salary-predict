# C:/Users/College/.conda/envs/myml/python.exe python

import argparse
import os
import config
from config import config_dict
import pandas as pd
import numpy as np
from time import time

if __name__ == "__main__":

    pt_ = time()*1000.0
    print("Cleaning Data...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", type = str)
    parser.add_argument("--ofile", type = str)
    args = parser.parse_args()

    df = pd.read_csv(config_dict[str(args.ifile)], header = None, names = config.COL_NAMES)
    
    # drop columns with missing values
    df = df.replace([config.UNK_TOKEN],value = np.nan).dropna()

    # modify target column and rename/create and drop
    df[config.TARGET_COLUMN] = df[config.INIT_TARGET].apply(lambda x: 1 if x in [' >50K', ' >50K.'] else 0)  # majority class: 0, minority: 1
    df.drop(config.INIT_TARGET, 1 ,inplace = True)

    df.to_csv(config_dict[str(args.ofile)],index = None)  ## dont forget to track later in git

    print(f"Data is clean!         Time Taken : {time()*1000.0 - pt_}ms ")