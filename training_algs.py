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

def randomTrainSynthetic(train_set):
    C_synth, C_synth_train, C_synth_val, C_synth_test = createSyntheticC()

    return {'A' 'C_synth_test':C_synth_test}

def createSyntheticC():
    #create random numpy 10x100 through product of two matrices
    A_synth = np.random.randint(0,2,size=(1,10))
    B_synth = np.random.randint(0,2,size=(1,100))
    C_synth = np.dot(A_synth.T, B_synth)
    C_synth = C_synth.astype(float)

    #shuffle all possible loop indices
    loop_indices = np.where(~np.isnan(C_synth))
    loop_indices = np.random.permutation(\
        np.concatenate((loop_indices[0].reshape(-1, 1), loop_indices[1].reshape(-1, 1)), axis=1)\
    )

    C_synth_train = np.copy(C_synth)
    C_synth_val = np.copy(C_synth)
    C_synth_test = np.copy(C_synth)

    #loop through
    iters = 0
    for index in loop_indices.tolist():
        if iters<700:
            #create training set
            C_synth_val[index[0],index[1]] = np.nan
            C_synth_test[index[0],index[1]] = np.nan
        elif iters<900:
            #create val set
            C_synth_train[index[0],index[1]] = np.nan
            C_synth_test[index[0],index[1]] = np.nan
        else:
            #create test set
            C_synth_train[index[0],index[1]] = np.nan
            C_synth_val[index[0],index[1]] = np.nan
        #iterate
        iters+=1
    pdb.set_trace()
    return C_synth, C_synth_train, C_synth_val, C_synth_test

def SyntheticTrain(train_set, rand_init, feature_size
, lam, learn_rate, max_iter, tolerance):
    lam_i = lam
    lam_j = lam

    #create the synthetic dataset
    C_synth, C_synth_train, C_synth_val, C_synth_test = createSyntheticC()

    #create I matrices that represents where there is data (Boolean)
    I = ~np.isnan(C_synth_train)
    I_val = ~np.isnan(C_synth_val)

    #convert all nans in C to 0, I and I_val will filter them!
    # C_synth[np.isnan(C_synth)] = 0.0
    # C_synth_val[np.isnan(C_synth_val)] = 0.0

    #initialize A and B which represents students and steps as (feature_size x students/steps)
    A = np.random.uniform(-rand_init, rand_init, size=(feature_size, C_synth.shape[0]))
    B = np.random.uniform(-rand_init, rand_init, size=(feature_size, C_synth.shape[1]))

    A, B = \
    StochasticGD_PMF(A, B, C_synth, C_synth_val, I, I_val, lam_i, lam_j, feature_size, learn_rate, max_iter, tolerance, True)

    return {'A':A, 'B':B, 'C_synth_test':C_synth_val}


def PMFTrain(train_set, val_set, original_data, rand_init, feature_size
, col_i, col_j, col_target, lam, learn_rate, max_iter, tolerance):
    '''IN PyMC3 example initialized all unkown values to gloabl average of known values'''
    lam_i = lam
    lam_j = lam

    #obtain sizes of the matrix C given original data
    unique_vals_i = original_data[col_i].unique().size
    unique_vals_j = original_data[col_j].unique().size

    #set global_avg of training data
    global_avg = train_set[col_target].mean()

    #create matrix C that contains the correct answers from train_set, matrix covers WHOLE original set
    #create this matrix for val/test set as well to track it's progress during optimization
    #{0,1} for correct or not, default_C for unkown datapoint
    C = PMFCreateC(original_data, col_i, col_j, col_target, np.nan, train_set)
    C_val = PMFCreateC(original_data, col_i, col_j, col_target, np.nan, val_set)

    #create I matrices that represents where there is data (Boolean)
    I = ~np.isnan(C)
    I_val = ~np.isnan(C_val)

    #convert all nans in C to 0, I and I_val will filter them!
    C[np.isnan(C)] = 0.0
    C_val[np.isnan(C_val)] = 0.0

    #initialize A and B which represents students and steps as (feature_size x students/steps)
    A = np.random.normal(loc=0.0, scale=rand_init, size=(feature_size, unique_vals_i))
    B = np.random.normal(loc=0.0, scale=rand_init, size=(feature_size, unique_vals_j))

    #optimize A and B
    A, B = StochasticGD_PMF(A, B, C, C_val, I, I_val, lam_i, lam_j, feature_size, learn_rate, max_iter, tolerance, True)

    return {'A':A, 'B':B, 'original_data':original_data,
    'col_i':col_i, 'col_j':col_j, 'col_target':col_target, 'default_C':np.nan}

def PMFLossFnc(A, B, C, lam_i, lam_j, feature_size):
    #non-regularization part of loss function
    #np.vectorize(dm.sigmoid)
    subtracter = (np.dot(A.T, B))
    diff = C - subtracter
    diff[np.isnan(diff)] = 0 #convert all diff nan's to 0's, 0's don't affect below calcs
    diff = 0.5*np.sum(diff**2)

    #add in regularization loss
    return diff + (lam_i/float(2))*np.sum(A**2) + (lam_j/float(2))*np.sum(B**2)

def PMFCreateC(original_data, col_i, col_j, col_target, default_C, sample_set):
    #create two dataframes with all unique values of col_i and col_j
    unique_i = pd.DataFrame(original_data[col_i].unique())
    unique_j = pd.DataFrame(original_data[col_j].unique())

    #make a new col 'key' to cross join by (hacky)
    unique_i['key'] = 0
    unique_j['key'] = 0

    #perform the cross join/cartiesian product
    original_data_cross = pd.merge(unique_i, unique_j, on=['key'], how='outer')
    original_data_cross = original_data_cross.drop('key', axis=1)
    original_data_cross.columns = [col_i, col_j]

    #now create C
    C = pd.merge(original_data_cross, sample_set[[col_i, col_j, col_target]]\
    , on=[col_i, col_j], how='left')

    # C[col_target] = C.apply(func=dm.SVDNullReplace, axis=1, args=(col_target+'_x', col_target+'_y', default_C))
    # C = C.drop([col_target+'_x', col_target+'_y'], axis=1)
    C = pd.pivot_table(C, values=col_target, index=col_i, columns=col_j\
    , aggfunc='sum').values

    C[np.isnan(C)] = default_C

    return C

def StochasticGD_PMF(A, B, C, C_val, I, I_val, lam_i, lam_j, feature_size, learn_rate, max_iter, tolerance, print_progress):
    #create loop indices for stochastic gradient descent that are shuffled, train on shuffled examples
    # loop_indices = np.where(~np.isnan(C))
    # loop_indices = np.random.permutation(\
    #     np.concatenate((loop_indices[0].reshape(-1, 1), loop_indices[1].reshape(-1, 1)), axis=1)\
    # )

    #initialize loss values to calc diffs
    loss = PMFLossFnc(A, B, C, lam_i, lam_j, feature_size)
    loss_val = PMFLossFnc(A, B, C_val, lam_i, lam_j, feature_size)

    #update A and B using gradients based on a random valid sample
    for iters in range(max_iter):
        #initilize gradients for all A(kxN) and B(kxM) elements
        grad_A = np.zeros(A.shape)
        grad_B = np.zeros(B.shape)

        #for all index[0] = k, index[1] = i and k,j update gradients
        for index, a in np.ndenumerate(A):
            # temp_dot_A= np.vectorize(dm.sigmoid)(np.dot(A[:,index[1]].T, B))
            grad_A[index] = np.sum(\
                I[index[1],:]*(C[index[1],:]-np.dot(A[:,index[1]].T, B))*\
                (-B[index[0],:])\
            ) + lam_i*A[index]

        for index, b in np.ndenumerate(B):
            # temp_sig = np.vectorize(dm.sigmoid)(np.dot(A.T, B[:,index[1]]))
            grad_B[index] = np.sum(\
                I[:,index[1]]*(C[:,index[1]]-np.dot(A.T, B[:,index[1]]))*\
                (-A[index[0],:])\
            ) + lam_j*B[index]

        A = A - learn_rate*grad_A
        B = B - learn_rate*grad_B

        # for i, j in loop_indices:
        #     #given an example from the training set encoded in C, calculate new gradients for A and B and update
        #     sig_dot = dm.sigmoid(np.dot(A[i,:], B[j,:].T))
        #
        #     grad_A = -(C[i,j]-sig_dot)*(sig_dot*(1-sig_dot))*(B[j,:]) + lam_i*A[i,:]
        #     grad_B = -(C[i,j]-sig_dot)*(sig_dot*(1-sig_dot))*(A[i,:]) + lam_j*B[j,:]
        #
        #     A[i,:] = A[i,:] - learn_rate*grad_A
        #     B[j,:] = B[j,:] - learn_rate*grad_B

        #check if tolerance has been met, (moved small enough to close enogh to local min)
        diff_loss = loss - PMFLossFnc(A, B, C, lam_i, lam_j, feature_size)
        diff_loss_val = loss_val - PMFLossFnc(A, B, C_val, lam_i, lam_j, feature_size)

        if print_progress:
            print('ITER: ', iters, ', DIFF_LOSS_TRAIN= ', diff_loss, 'DIFF_LOSS_VAL= ', diff_loss_val,
            ', CUR_LOSS_TRAIN= ', loss, 'CUR_LOSS_VAL= ', loss_val)

        #stop when under tolerance OR valadation set starts increasing in loss
        if ((diff_loss < tolerance) or ((diff_loss_val < 0) and (iters>0.1*max_iter))):
            break
        loss = PMFLossFnc(A, B, C, lam_i, lam_j, feature_size)
        loss_val = PMFLossFnc(A, B, C_val, lam_i, lam_j, feature_size)

    return A, B
