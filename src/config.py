# \.conda\envs\myml\python.exe python

# paths
CLEAN_DATA = "../input/train.csv"
FOLDED_DATA = "../input/training_folds.csv"

# column names 
KFOLD_COLUMN = 'k_val'
TARGET_COLUMN = 'target'
INIT_TARGET = '50k'
COL_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
            'hours-per-week', 'native-country', '50k']
CAT_COLUMNS = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
              'sex', 'native-country']
CONT_COLUMNS = ['age','fnlwgt','hours-per-week', 'capital-gain', 'capital-loss']
