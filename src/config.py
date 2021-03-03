# \.conda\envs\myml\python.exe python


# dictionary to store filepaths
config_dict = dict([])


# paths
config_dict['CLEAN_DATA'] = "../input/processed/train.csv"
config_dict['PROCESSED_DATA'] = "../input/processed/train_processed.csv"
config_dict['FOLDED_DATA'] = "../input/interim/training_folds.csv"
config_dict['RAW_DATA'] = "../input/raw/adult.data"


# column names 
KFOLD_COLUMN = 'k_val'
TARGET_COLUMN = 'target'
INIT_TARGET = '50k'
COL_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k']
CAT_COLUMNS = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
CONT_COLUMNS = ['age','fnlwgt','hours-per-week', 'capital-gain', 'capital-loss']


# other variables
UNK_TOKEN = ' ?'
mean_encoding = []
MEAN_ENCODED_COLUMN = '_mean_enc'
RANDOM_STATE = 42
NUM_FOLDS = 6