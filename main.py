import pdb
import numpy as np
import scipy as sp
import pandas as pd

import data_manip as dm
import testing_functions as tf
import run_for_results as run

import training_algs as ta
import predict_algs as pa
import performance_metrics as pm

import random
import time

'''FIRST CREATE TRAIN, VAL, & TEST SETS, THEN SAVE THEM FOR TESTING/PREDICTION, COMMENT OUT SECTION AFTER'''
# t0 = time.time()
# raw_train = dm.importText('/Users/qipanda/Documents/2016-2017_Thesis/education_data/\
# bridge_to_algebra_2008_2009/bridge_to_algebra_2008_2009_train.txt',\
# '\t', False, 100000)
# import pdb; pdb.set_trace()
# # raw_train = dm.importText('/Users/qipanda/Documents/2016-2017_Thesis/education_data/\
# # bridge_to_algebra_2008_2009/bridge_test.txt',\
# # '\t', True, 1000)
# print('in main, data loaded at: ', (time.time()-t0))
#
# train_set_original = dm.splitCol(raw_train, 'Problem Hierarchy', ', ', ['Unit', 'Section'])
# print('in main, data split at: ', (time.time()-t0))
#
# train_set_original = dm.changeColsToDtime(train_set_original, ['Step Start Time','First Transaction Time',\
# 'Correct Transaction Time','Step End Time'])
# print('in main, data modified at: ', (time.time()-t0))
#
# train_set_original_merged = dm.mergeCols(dataset=train_set_original\
# , cols_to_merge=['Unit', 'Section', 'Problem Name', 'Step Name', 'Problem View', 'KC(SubSkills)', 'KC(KTracedSkills)'], delim=', '\
# , new_col_name='Step Hierarchy', drop_old=False)
# print('in main, data merged at: ', (time.time()-t0))
#
# createValParameters = {
#                         'dataset':train_set_original_merged,
#                         'max_iter': 10,
#                         'prct_of_data': 0.1,
#                         'problem_thresh_prct': 0.5,
#                         'problem_thresh': 2
#                         }
#
# train_set, val_set, test_set, rows_retained, rows_lost, prct_lost, prct_ratio_val_to_train = \
# dm.createValidationSet(**createValParameters)
# print('DATA PARTITION COMPLETE, prct_lost: ', prct_lost, ' prct_ratio_val_to_train: ', prct_ratio_val_to_train)
#
# #now save each dataset into files,
# train_set.to_pickle('train_set.pkl')
# val_set.to_pickle('val_set.pkl')
# test_set.to_pickle('test_set.pkl')
# train_set_original_merged.to_pickle('original_set.pkl')
# print('DATA SAVED, TERMINATING')
# quit()

'''BELOW AFTER YOU HAVE CREATED TRAIN, VAL, & TEST SETS'''

train_set = pd.read_pickle('train_set.pkl')
val_set = pd.read_pickle('val_set.pkl')
test_set = pd.read_pickle('test_set.pkl')
train_set_original_merged = pd.read_pickle('original_set.pkl')

train_params = {
                'val_set':val_set,
                'original_data':train_set_original_merged,
                'rand_init':1,
                'feature_size':5,
                'col_i':'Anon Student Id',
                'col_j':'Step Hierarchy',
                'col_target':'Correct First Attempt',
                'lam':0,
                'learn_rate':1e-4,
                'max_iter':int(1e2),
                'tolerance':1e-12
                }

# train_params = {
#                 'rand_init':1e-1,
#                 'feature_size':1,
#                 'lam':0.74,
#                 'learn_rate':0.06,
#                 'max_iter':int(1e4),
#                 'tolerance':1e-18
#                 }

hypertest_params = {
                    'train_set':train_set,
                    'val_set':val_set,
                    'train_alg':ta.PMFTrain,
                    'train_params':train_params,
                    'target_param':'feature_size',
                    'increments':1,
                    'iters_per_inc':1,
                    'print_progress':True,
                    'mult':True,
                    'base':10,
                    'pred_alg':pa.PMFPred,
                    'performance_algs':[pm.calcRMSE, pm.calcBinaryPred]
                    }

hyperparams, results = run.hyperParamTesting(**hypertest_params)
print('HYPERPARAM TESTING COMPLETE: RESULTS: ',
'\nHYPERPARAMS: ', hyperparams,
'\nRESULTS: ', results )
import pdb; pdb.set_trace()
quit()

''' ONE DETERMINED HYPERPARAMS TEST FINAL RESULTS HERE'''
# train_params = {
#                 'val_set':test_set,
#                 'original_data':train_set_original_merged,
#                 'rand_init':1e-3,
#                 'feature_size':5,
#                 'default_C':np.nan,
#                 'col_i':'Anon Student Id',
#                 'col_j':'Step Hierarchy',
#                 'col_target':'Correct First Attempt',
#                 'lam':1e-3,
#                 'learn_rate':1e-3,
#                 'max_iter':int(1e3),
#                 'tolerance':1e-9
#                 }

train_params = {
                'rand_init':1e-1,
                'feature_size':1,
                'lam':0.74,
                'learn_rate':0.06,
                'max_iter':int(1e4),
                'tolerance':1e-18
                }

finaltest_params = {
                    'train_set':train_set,
                    'test_set':test_set,
                    'train_alg':ta.SyntheticTrain,
                    'train_params':train_params,
                    'iters':500,
                    'pred_alg':pa.PMFPredSynthetic,
                    'performance_algs':[pm.calcRMSE, pm.calcBinaryPred]
}

#test results: array([ 0.4999881 ,  0.52181818])
#learning NOTHING!

avg_test_results, test_results = run.finalTesting(**finaltest_params)
print('TESTSET DONE, RESULTS: ', test_results)
print('AVG_RESULTS: ', avg_test_results)
pdb.set_trace()

print('HI')
