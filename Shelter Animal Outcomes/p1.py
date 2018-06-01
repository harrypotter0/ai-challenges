import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import sys
from datetime import datetime

if __name__ == "__main__":
    in_file_train = '../input/train.csv'
    in_file_test = '../input/test.csv'

    print("Loading data...\n")
    pd_train = pd.read_csv(in_file_train)
    pd_test = pd.read_csv(in_file_test)

    print("Munging data...\n")
    pd_train = munge(pd_train,True)
    pd_test = munge(pd_test,False)
