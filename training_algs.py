import pdb #debugger
import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import random
import math
import data_manip as dm

def randomTrain(train_set, val_set):
    return {'pred_col_name':'Pred', 'original_col_name':'Correct First Attempt', 'predLB':0, 'predUB':1}

def PMFTrain(train_set, val_set, original_data, LB_rand_init, UB_rand_init, feature_size
, default_C, col_i, col_j, col_target, lam_i, lam_j, learn_rate, max_iter, tolerance):
    #obtain sizes of the matrix C
    unique_vals_i = original_data[col_i].unique().size
    unique_vals_j = original_data[col_j].unique().size

    #create matrix C that contains the correct answers from train_set, matrix covers WHOLE original set
    #{0,1} for correct or not, nan for unkown datapoint
    #_x is the left dup col, _y is the right dup col
    C = PMFCreateC(original_data, col_i, col_j, col_target, default_C, train_set)

    #create val_C to monitor Loss of valadation set as we iter
    C_val = PMFCreateC(original_data, col_i, col_j, col_target, default_C, val_set)

    #create x_init as a long vector to pass into the optimizer that will be reconstructed into A and B
    #optimizer does not take matrices...
    A = np.random.uniform(LB_rand_init, UB_rand_init, size=(unique_vals_i, feature_size))
    B = np.random.uniform(LB_rand_init, UB_rand_init, size=(unique_vals_j, feature_size))

    A, B = StochasticGD_PMF(A, B, C, C_val, lam_i, lam_j, feature_size, learn_rate, max_iter, tolerance, True)

    return {'A':A, 'B':B, 'original_data':original_data,
    'col_i':col_i, 'col_j':col_j, 'col_target':col_target, 'default_C':default_C}

def PMFLossFnc(A, B, C, lam_i, lam_j, feature_size):
    #non-regularization part of loss function
    subtracter = np.vectorize(dm.sigmoid)(np.dot(A, np.transpose(B)))
    diff = C - subtracter
    diff[np.isnan(diff)] = 0 #convert all diff nan's to 0's, 0's don't affect below calcs
    diff = 0.5*np.sum(diff**2)

    #add in regularization loss
    return diff + (lam_i/float(2))*np.sum(A**2) + (lam_j/float(2))*np.sum(B**2)

def PMFCreateC(original_data, col_i, col_j, col_target, default_C, sample_set):
    C = pd.merge(original_data[[col_i, col_j, col_target]], sample_set[[col_i, col_j, col_target]]\
    , on=[col_i, col_j], how='left')
    C[col_target] = C.apply(func=dm.SVDNullReplace, axis=1, args=(col_target+'_x', col_target+'_y', default_C))
    C = C.drop([col_target+'_x', col_target+'_y'], axis=1)
    C = pd.pivot_table(C, values=col_target, index=col_i, columns=col_j\
    , aggfunc='sum').values

    return C

def StochasticGD_PMF(A, B, C, C_val, lam_i, lam_j, feature_size, learn_rate, max_iter, tolerance, print_progress):
    #create loop indices for stochastic gradient descent that are shuffled

    loop_indices = np.where(~np.isnan(C))
    loop_indices = np.random.permutation(\
        np.concatenate((loop_indices[0].reshape(-1, 1), loop_indices[1].reshape(-1, 1)), axis=1)\
    )

    loss = PMFLossFnc(A, B, C, lam_i, lam_j, feature_size)
    loss_val = PMFLossFnc(A, B, C_val, lam_i, lam_j, feature_size)

    #update A and B using gradients based on a random valid sample
    for iters in range(max_iter):

        for i, j in loop_indices:
            sig_dot = dm.sigmoid(np.dot(A[i,:], B[j,:].T))

            grad_A = -(C[i,j]-sig_dot)*(sig_dot*(1-sig_dot))*(B[j,:]) + lam_i*A[i,:]
            grad_B = -(C[i,j]-sig_dot)*(sig_dot*(1-sig_dot))*(A[i,:]) + lam_j*B[j,:]

            A[i,:] = A[i,:] - learn_rate*grad_A
            B[j,:] = B[j,:] - learn_rate*grad_B

        #check if tolerance has been met, (moved small enough to close enogh to local min)
        diff_loss = loss - PMFLossFnc(A, B, C, lam_i, lam_j, feature_size)
        diff_loss_val = loss_val - PMFLossFnc(A, B, C_val, lam_i, lam_j, feature_size)

        if (diff_loss < tolerance):
            break
        loss = PMFLossFnc(A, B, C, lam_i, lam_j, feature_size)
        loss_val = PMFLossFnc(A, B, C_val, lam_i, lam_j, feature_size)

        if print_progress:
            print('ITER: ', iters, ', DIFF_LOSS_TRAIN= ', diff_loss, 'DIFF_LOSS_VAL= ', diff_loss_val,
            ', CUR_LOSS_TRAIN= ', loss, 'CUR_LOSS_VAL= ', loss_val)

    return A, B
