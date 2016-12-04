import pdb #debugger
import pandas as pd
import numpy as np
import random
import math

import data_manip as dm
import training_algs as ta

def randomPred(val_set, original_col_name, pred_col_name, predLB, predUB):
    #return the correct values and prediction values as numpy arrays
    #pred_vals applies func1d(a, *args) elementwise
    original_vals = val_set[original_col_name].as_matrix().reshape((val_set.shape[0]), 1)
    pred_vals = np.apply_along_axis(func1d=random.uniform, axis=1, \
    arr=np.zeros((val_set.shape[0], 1)), b=1)

    return original_vals, pred_vals

def PMFPred(val_set, A, B, original_data, col_i, col_j, col_target, default_C):
    #construct prediction C
    #np.vectorize(dm.sigmoid)()
    pred_C = np.dot(A.T, B)

    C = ta.PMFCreateC(original_data, col_i, col_j, col_target, default_C, val_set)

    # #construct actual C of valadation set (same as in training but with val_data)
    # C = pd.merge(original_data[[col_i, col_j, col_target]], val_set[[col_i, col_j, col_target]]\
    # , on=[col_i, col_j], how='left')
    # C[col_target] = C.apply(func=dm.SVDNullReplace, axis=1, args=(col_target+'_x', col_target+'_y', default_C))
    # C = C.drop([col_target+'_x', col_target+'_y'], axis=1)
    # C = pd.pivot_table(C, values=col_target, index=col_i, columns=col_j\
    # , aggfunc='sum').values

    #return vectors of prediction from A*B and C_val to calc error metrics
    original_vals = C[~np.isnan(C)].reshape(-1, 1)
    pred_vals = pred_C[~np.isnan(C)].reshape(-1, 1)

    return original_vals, pred_vals

def PMFPredSynthetic(val_set, A, B, C_synth_test):
    #construct prediction C
    #np.vectorize(dm.sigmoid)()
    pdb.set_trace()
    pred_C = np.dot(A.T, B)

    #return vectors of prediction from A*B and C_val to calc error metrics
    original_vals = C_synth_test[~np.isnan(C_synth_test)].reshape(-1, 1)
    pred_vals = pred_C[~np.isnan(C_synth_test)].reshape(-1, 1)

    return original_vals, pred_vals

#for iterating through val_set
    # for i, row in val_set.iterrows():
    #     pred_vals[i] = random.uniform(predLB, predUB)
    #     print('Finished randomPred prediction ', i)
