from config import COL_NAMES, CAT_COLUMNS, CONT_COLUMNS, TARGET_COLUMN, config_dict
from sklearn.pipeline import Pipeline
import pandas as pd
import tools
import argparse
from time import time

def remove_items(l_ = [], d_ = []):
    return list(filter(lambda x: x not in d_, l_))

if __name__ == "__main__":
    
    pt_ = time()*1000.0
    print("Starting Feature Engineering...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", type = str)
    parser.add_argument("--ofile", type = str)
    args = parser.parse_args()

    ddf = pd.read_csv(config_dict[str(args.ifile)])

    pipe = Pipeline([
         ("cn", tools.CustomNormalizer(ntype = "sc", column_array = remove_items(CONT_COLUMNS,['capital-gain', 'capital-loss']))),
         ("cme", tools.CustomMeanEncoder([TARGET_COLUMN], remove_items(CAT_COLUMNS, ['education']))),
         ("drop", tools.DropColumns(['education', 'capital-gain', 'capital-loss']) ),
         ("ohe", tools.CustomOneHotEncoder(remove_items(CAT_COLUMNS, ['education','sex']), ['sex']))])

    ddf = pipe.fit_transform(ddf)
    
    ddf.to_csv(config_dict[str(args.ofile)],index = None)
    
    print(f"Feature Engineering Completed!         Time Taken : {time()*1000.0 - pt_}ms ")


