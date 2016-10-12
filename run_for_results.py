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

def predictWithVal(dataset, max_iter, prct_of_data, problem_thresh_prct, problem_thresh, \
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
        = dm.createValidationSet(dataset, max_iter, prct_of_data, problem_thresh_prct, problem_thresh)
        print('iter= ', i, ', prct_lost= ', prct_lost, ', prct_ratio_actl= ', prct_ratio_actl)

        #retrn a dictionary of params from train_set datagrame, train_alg fnc, dict of train_params for train_alg
        if train_params == None:
            val_params = train_alg(train_set)
        else:
            val_params = train_alg(train_set, **train_params)

        if val_params == None:
            original_vals, pred_vals = pred_alg(val_set)
        else:
            original_vals, pred_vals = pred_alg(val_set, **val_params)

        #list of performance summary metrics given the predictions for each performance_alg in the list
        performance_metrics[i] = calculatePerformance(original_vals, pred_vals, performance_algs)

    return performance_metrics

def calculatePerformance(original_vals, pred_vals, performance_algs):
    #iterate through all performance metrics that we want to calculate for a given validation set
    results = np.zeros(shape=len(performance_algs))

    for k, performance_calc_fnc in enumerate(performance_algs):
        results[k] = performance_calc_fnc(original_vals, pred_vals)

    return results
