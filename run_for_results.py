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

def hyperParamTesting(train_set, val_set,\
train_alg, train_params,\
target_param, increments, iters_per_inc, print_progress, mult, base,\
pred_alg, performance_algs):

    '''
    This algorithm is designed to find good hyperparameters for a given algorithm

    i is for every perfmutation/increment of said hyperparameter
    j is how many times we will run each increment and then average the performances (differences in rand init)
    initialize the results array, +1 to store the target_hyperparam values
    '''

    results = np.zeros((increments, iters_per_inc, len(performance_algs)))
    hyperparams = np.zeros(increments)

    for i in range(increments):
        #label the hyperparam we are iterating over
        hyperparams[i] = train_params[target_param]

        for j in range(iters_per_inc):
            #train the paramaters for prediction
            learned_params = train_alg(train_set, **train_params)

            #return two vectors of predictions
            original_vals, pred_vals = pred_alg(val_set, **learned_params)

            #calculate the performance(s) on the val_set and log it
            results[i,j] = calculatePerformance(original_vals, pred_vals, performance_algs)

        #increment the target_param in train_params either mult or addition
        if mult:
            train_params[target_param] *= base
        else:
            train_params[target_param] += base

    #average across every iters_per_inc for each increment on both performance_algs
    avg_results = np.average(results, axis=1)

    return hyperparams, avg_results

def finalTesting(train_set, test_set,\
train_alg, train_params, \
iters,\
pred_alg, performance_algs):

    '''This algorithm is to perform training and evaluation on the test_set after hyperparams have been decided'''

    results = np.zeros((iters, len(performance_algs)))
    for i in range(iters):
        #train the paramaters for prediction
        learned_params = train_alg(train_set, **train_params)

        #return two vectors of predictions
        original_vals, pred_vals = pred_alg(test_set, **learned_params)

        #calculate the performance(s) on the val_set and log it
        results[i] = calculatePerformance(original_vals, pred_vals, performance_algs)

    avg_results = np.average(results, axis=0)

    return avg_results, results

def calculatePerformance(original_vals, pred_vals, performance_algs):

    '''iterate through all performance metrics that we want to calculate for a given validation set'''
    results = np.zeros(shape=len(performance_algs))

    for k, performance_calc_fnc in enumerate(performance_algs):
        results[k] = performance_calc_fnc(original_vals, pred_vals)

    return results


'''FUNCTIONS BELOW ARE DEPRECIATED'''

def predictWithVal(dataset, valCreationParams, \
val_iter, train_alg, train_params, pred_alg, performance_algs):
    #this function takes any predictor method and obtains model parameters using
    #the average of random cross-validation sets. Cross-validation method will be
    #leave prct_of_data% of dataset out as the val set, repeat this randomly val_iter times
    #each val_iter will re-randomly select from the original dataset
    #(time sensitive data makes it hard to apply cross-folds validation)

    #performance_algs is a list of performance_metric functions to apply to results
    performance_metrics = np.empty((val_iter, len(performance_algs)))

    for i in range(val_iter):
        #create val and train set for this val_iter
        val_set, train_set, rows_retained, rows_lost, prct_lost, prct_ratio_actl\
        = dm.createValidationSet(dataset, **valCreationParams)
        print('IN RUN FOR RESULTS: ', 'iter= ', i, ', prct_lost= ', prct_lost, ', prct_ratio_actl= ', prct_ratio_actl)
        pdb.set_trace()
        #retrn a dictionary of params from train_set datagrame, train_alg fnc, dict of train_params for train_alg
        if train_params == None:
            val_params = train_alg(train_set, val_set)
        else:
            val_params = train_alg(train_set, val_set, **train_params)

        if val_params == None:
            original_vals, pred_vals = pred_alg(val_set)
        else:
            original_vals, pred_vals = pred_alg(val_set, **val_params)

        #list of performance summary metrics given the predictions for each performance_alg in the list
        performance_metrics[i] = calculatePerformance(original_vals, pred_vals, performance_algs)

    return np.average(performance_metrics, axis=0)
