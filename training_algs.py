import pdb #debugger
import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import random

import data_manip as dm

def randomTrain(train_set):
    return {'pred_col_name':'Pred', 'original_col_name':'Correct First Attempt', 'predLB':0, 'predUB':1}

def SVDTrain(train_set, original_data, LB_rand_init, UB_rand_init, feature_size, default_C, col_i, col_j, col_target):
    pdb.set_trace()
    #obtain sizes of the matrix C
    unique_vals_i = original_data[col_i].unique().size
    unique_vals_j = original_data[col_j].unique().size

    #create matrix C that contains the correct answers from train_set, matrix covers WHOLE original set
    #{0,1} for correct or not, nan for unkown datapoint
    #_x is the left dup col, _y is the right dup col
    C = pd.merge(original_data[[col_i, col_j, col_target]], train_set[[col_i, col_j, col_target]]\
    , on=[col_i, col_j], how='left')
    C[col_target] = C.apply(func=dm.SVDNullReplace, axis=1, args=(col_target+'_x', col_target+'_y', default_C))
    C = C.drop([col_target+'_x', col_target+'_y'], axis=1)
    C = pd.pivot_table(C, values=col_target, index=col_i, columns=col_j\
    , aggfunc='sum').values

    #create x_init as a long vector to pass into the optimizer that will be reconstructed into A and B
    #optimizer does not take matrices...
    x_init = \
    np.random.uniform(LB_rand_init, UB_rand_init, size=(unique_vals_i*feature_size + unique_vals_j*feature_size))

    #minimize ||C - AB'||_F^2 by A and B (the Frobenius norm)
    result = sp.optimize.minimize(fun=SVDLossFnc, x0=x_init, args=(C, unique_vals_i, unique_vals_j, feature_size), options={'maxiter':10, 'disp':True})
    pdb.set_trace()

def SVDLossFnc(x, C, i_size, j_size, feature_size):
    #reconstruct A, B, and C from x
    A = x[0:i_size*feature_size].reshape(i_size, feature_size)
    B = x[i_size*feature_size:].reshape(j_size, feature_size)

    #base calculations of norm arguments
    subtracter = np.dot(A, np.transpose(B))
    diff = C - subtracter

    #convert all diff nan's to 0's, 0's don't affect norm calculation
    diff[np.isnan(diff)] = 0

    #return Frobenius norm
    return np.linalg.norm(diff, ord='fro')
