import pdb
import numpy as np
import scipy as sp

import data_manip as dm
import testing_functions as tf
import run_for_results as run

import training_algs as ta
import predict_algs as pa
import performance_metrics as pm

import random
import time

# #this code is to see what the test data is like
# pdb.set_trace()
# test_data = raw_train = dm.importText('/Users/qipanda/Documents/2016-2017_Thesis/education_data/\
# bridge_to_algebra_2008_2009/bridge_to_algebra_2008_2009_test.txt',\
# '\t', False, 1000)
# pdb.set_trace()

#import the training and test sets
t0 = time.time()
raw_train = dm.importText('/Users/qipanda/Documents/2016-2017_Thesis/education_data/\
bridge_to_algebra_2008_2009/bridge_to_algebra_2008_2009_train.txt',\
'\t', False, 100000)
print('in main, data loaded at: ', (time.time()-t0))

train_set_original = dm.splitCol(raw_train, 'Problem Hierarchy', ', ', ['Unit', 'Section'])
print('in main, data split at: ', (time.time()-t0))

train_set_original = dm.changeColsToDtime(train_set_original, ['Step Start Time','First Transaction Time',\
'Correct Transaction Time','Step End Time'])
print('in main, data modified at: ', (time.time()-t0))

train_set_original_merged = dm.mergeCols(dataset=train_set_original\
, cols_to_merge=['Problem View', 'Unit', 'Section', 'Problem Name', 'Step Name'], delim=', '\
, new_col_name='Step Hierarchy', drop_old=False)
print('in main, data merged at: ', (time.time()-t0))

# pdb.set_trace()
# performance_metrics_PMF = run.predictWithVal(train_set_original_merged, 100, 0.40, 0.50, 2\
# , 1, ta.PMFTrain\
# , {'original_data': train_set_original_merged, 'LB_rand_init':-0.001, 'UB_rand_init':0.001\
# , 'feature_size':10, 'default_C':np.nan, 'col_i':'Anon Student Id', 'col_j':'Step Hierarchy'\
# , 'col_target':'Correct First Attempt', 'lam_i':0.1, 'lam_j':0.1, 'learn_rate':0.1\
# , 'max_iter':1000, 'tolerance':1e-8}, pa.PMFPred, [pm.calcRMSE])
# print(performance_metrics_PMF)

performance_metrics_random = run.predictWithVal(train_set_original_merged, 100, 0.40, 0.50, 2\
, 1, ta.randomTrain\
,None, pa.randomPred, [pm.calcRMSE])
print(performance_metrics_random)
pdb.set_trace()
#this code is for testing params for valadation splitting
# pdb.set_trace()
# results = tf.testFncIter(20, 200, [0, 0, 0, 0.025, 0], [False, False, True, True, True], [2,3,4,5], [np.average, np.min, np.max]\
# , dm.createValidationSet, train_set_original, 100, 0.2, 0., 2.)
# pdb.set_trace()
# val_set, train_set, rows_retained, rows_lost, prct_lost, prct_ratio_actl \
# = dm.createValidationSet(dataset=train_set_original, max_iter=100\
# , prct_of_data=0.40, problem_thresh_prct=0.50, problem_thresh=2)
