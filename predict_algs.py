import pdb #debugger
import pandas as pd
import numpy as np
import random

def randomPred(val_set, original_col_name, pred_col_name, predLB, predUB):
    #return the correct values and prediction values as numpy arrays
    #pred_vals applies func1d(a, *args) elementwise
    original_vals = val_set[original_col_name].as_matrix().reshape((val_set.shape[0]), 1)
    pred_vals = np.apply_along_axis(func1d=random.uniform, axis=1, \
    arr=np.zeros((val_set.shape[0], 1)), b=1)

    return original_vals, pred_vals


#for iterating through val_set
    # for i, row in val_set.iterrows():
    #     pred_vals[i] = random.uniform(predLB, predUB)
    #     print('Finished randomPred prediction ', i)
