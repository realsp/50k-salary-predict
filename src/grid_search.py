from catboost import Pool
from catboost import CatBoost
from config import config_dict, TARGET_COLUMN
import pandas as pd
import matplotlib.pyplot as plt
from time import time

pt_ = time()
print("Starting Grid Search Algo for catboost...")

param_grid = {'learning_rate' : [0.001, 0.01, 0.1], 
              'l2_leaf_reg' : [0.1, 1, 3, 10], 
              'depth' : [3, 6, 9, 12],
              'iterations' : [100, 1000, 3000, 7000, 11000]}

model = CatBoost()
X = pd.read_csv(config_dict['PROCESSED_DATA'])
y = X[TARGET_COLUMN]
X = Pool(X.drop(TARGET_COLUMN, 1), y)

grid_search_result = model.grid_search(param_grid,
                                        X,
                                        cv=5,
                                        partition_random_seed=0,
                                        calc_cv_statistics=True,
                                        stratified=True,
                                        train_size=0.8,
                                        verbose=True,
                                        plot=True)

pd.DataFrame.from_dict(grid_search_result).to_csv('grid_params.csv',index=None)

print(f"Parameter Search Completed!    Time Taken : {time()-pt_}s")
print("Best params found:")
print(grid_search_result['params'])