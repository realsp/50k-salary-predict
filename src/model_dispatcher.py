from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

rfe_estimators = {'decision_tree': DecisionTreeClassifier(**{'class_weight':'balanced'}), 
'logistic_reg' : LogisticRegression(**{'verbose':1, 'solver':'liblinear', 'penalty':'l1', 'class_weight':'balanced', 'max_iter':1000})}

# 3 models, random forests, gbdt and logistic regression 

train_estimators = {}

catboost_params = {"parameters":[{"depth":9,"learning_rate":0.001,"l2_leaf_reg":10,"iterations":11000}
