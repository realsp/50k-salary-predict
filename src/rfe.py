from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from model_dispatcher import rfe_estimators
from config import config_dict, TARGET_COLUMN
import pandas as pd
from matplotlib import pyplot as plt
from time import time

if __name__ == "__main__":

      pt_ = time()
      print("Starting Recursive Feature Elimination...")

      df = pd.read_csv(config_dict['PROCESSED_DATA'])
      X = df.drop(TARGET_COLUMN,1)
      y = df[TARGET_COLUMN]

      feature_ranking = []

      for model in estimators.keys():
            min_features_to_select = 1  # Minimum number of features to consider
            rfecv = RFECV(estimator=estimators[model], step=1, cv=StratifiedKFold(5), scoring='f1', min_features_to_select = min_features_to_select, verbose = 1)
            rfecv.fit(X, y)

            print("Optimal number of features : %d" % rfecv.n_features_)

            feature_ranking.append(rfecv.ranking_)

            # Plot number of features VS. cross-validation scores

            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation f1 score - " + model)
            plt.plot(range(min_features_to_select,
                        len(rfecv.grid_scores_) + min_features_to_select),
                  rfecv.grid_scores_)
            plt.show()


      feature_df = pd.DataFrame({'col':feature_ranking})
      feature_df.to_csv("../feature_df.csv")

      print(f"Recursive Feature Elimination Completed!         Time Taken : {time() - pt_}s ")


