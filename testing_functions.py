import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb #debugger

def testFncIter(iters, iters_per_inc, arg_inc, args_to_inc, results_to_take, agg_fncs, fnc, *args):
    #return summary metrics about function performance over iter increments
    #test each iter iter_per_inc times for summary metrics
    #arg_inc is an array size of args that tells how much to increment each arg every iter
    #results list that indicates indices of which results to aggregate and analyze form fnc
    #agg_fncs is a list of functions to utilize in aggregation
    #don't increment on the first iter

    #initialize numpy 3darray to store results labels and results,
    #k is every agg_type, i is every iter_combination, j is is every variable
    #is hardcoded for the 3 metrics we are using
    results = np.zeros(shape=(len(agg_fncs), iters, len(results_to_take)))

    for i in range(iters):
        #initialize this iter combination of args and storage of this iters results
        #every jth row in iter_data is a sample, want to aggregate by collapsing rows
        #make sure to only increment the args from args_to_inc using loop below

        iter_args = [None]*len(args)
        for index, cur_arg in enumerate(args):
            if args_to_inc[index]:
                iter_args[index] = cur_arg + i*arg_inc[index]
            else:
                iter_args[index] = cur_arg

        iter_data = np.zeros(shape=(iters_per_inc, len(results_to_take)))

        #input results of every iter_per_ic, * seperates list of args for function to use
        for j in range(iters_per_inc):

            #make sure that only the results_to_take indexed results are aggregated
            iter_instance = fnc(*iter_args)
            iter_instance_filtered = []
            for index in results_to_take:
                iter_instance_filtered.append(iter_instance[index])

            iter_data[j] = iter_instance_filtered

        #calculate average, max, min of each result (assuming numpy aggregation with axis)
        for k, agg_fnc in enumerate(agg_fncs):
            results[k,i] = agg_fnc(iter_data, axis=0)

    return results
