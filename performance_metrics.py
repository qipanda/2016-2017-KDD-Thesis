import pdb #debugger
import pandas as pd
import numpy as np
import random

def calcRMSE(original_vals, pred_vals):
    #append two columns together along the columns
    appendCols = np.append(pred_vals, original_vals, axis=1)
    RMSE = 0
    for i in np.arange(appendCols.shape[0]):
        RMSE += (appendCols[i,0] - appendCols[i,1])**2

    RMSE /= appendCols.shape[0]
    RMSE = RMSE**(0.5)

    return RMSE

# #for testing purposes only
# def calcRMSE2(original_vals, pred_vals):
#     #append two columns together along the columns
#     appendCols = np.append(pred_vals, original_vals, axis=1)
#     RMSE = 0
#     for i in np.arange(appendCols.shape[0]):
#         RMSE += (appendCols[i,0] - appendCols[i,1])**2
#
#     RMSE /= appendCols.shape[0]
#     RMSE = RMSE**(0.5)
#
#     return 2*RMSE
