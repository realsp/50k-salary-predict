from catboost import Pool, CatBoostClassifier
from config import config_dict, NUM_FOLDS, KFOLD_COLUMN, TARGET_COLUMN
import pandas as pandas

best_parameters = {"depth":9,"learning_rate":0.001,"l2_leaf_reg":10,"iterations":11000}
# from hyperparameter tuning : (177th iteration; refer to line 181 in ../src/catboost_training.json)

X = pd.read_csv(config_dict['PROCESSED_DATA'])

X_train = X.drop(TARGET_COLUMN,1)
y_train = X[TARGET_COLUMN]
cbc = CatBoostClassifier(**best_parameters)
cbc.fit(X_train, y_train)
cbc.save_model("../models/model0.cbm")