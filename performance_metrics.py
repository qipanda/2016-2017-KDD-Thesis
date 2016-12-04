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

def calcBinaryPred(original_vals, pred_vals):
    #convert pred vals to be 1 when >threshhold, 0 else
    pred_vals = pred_vals>0.5
    correct = original_vals==pred_vals

    return np.sum(correct)/correct.shape[0]
