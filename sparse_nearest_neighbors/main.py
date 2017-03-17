import DataManipulators
import Learners
import loading_util as lu
import graphing_util as gu

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
'''
this script is meant to show the results of testing on single KC data
'''

def load_and_assign_tm():
    '''Prelim A: Load single KC data'''
    single_KC_data = pd.read_pickle('/Users/qipanda/Documents/2016-2017_KDD_Thesis/education_data'+\
        '/bridge_to_algebra_2008_2009/bridge_0809_KC.pkl')

    dm = DataManipulators.DataManipulator(df=single_KC_data)

    '''Prelim B: assign tm_ranks to the data'''
    assign_rel_tm_params = {
        'granularity_grp':  ['Anon Student Id', 'Problem Name', 'Problem View'],
        'time_within_col':  'Anon Student Id',
        'time_col':         'First Transaction Time',
        'tm_order_col':     'First Transaction Time Order'
    }
    dm.assign_rel_tm(**assign_rel_tm_params)

    return dm

def load_test_df():
    test_df = pd.DataFrame(
        {
            'Anon Student Id':['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
            'Problem Name':['P1', 'P1', 'P2', 'P2', 'P1', 'P1', 'P2', 'P2', 'P1', 'P1', 'P1', 'P1', 'P2'],
            'Step Name':['S1', 'S1', 'S1', 'S2', 'S2', 'S1', 'S3', 'S3', 'S1', 'S1', 'S1', 'S1', 'S3'],
            # 'First Transaction Time Rank':[1, 1, 2, 1, 2, 1, 2, 1, 2, 3],
            'First Transaction Time':['2008-09-01', '2008-09-02', '2008-09-03', '2008-09-04',\
                '2008-09-05', '2008-09-01', '2008-09-02', '2008-09-03', '2008-09-01', '2008-09-02',\
                '2008-09-03', '2008-09-04', '2008-09-05'],
            'Correct First Attempt':[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            'Problem View':[1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 3, 4, 1]
        }
    )

    return test_df

def preprocess_ftrs(dm):
    '''Preprocess features in DM'''
    assign_2D_sparse_ftrs_params = {
        'ftr_name':     'X_correct',
        'sim_col':      'Anon Student Id',
        'ftr_cols':     ['Problem Name', 'Step Name'],
        'tm_order_col': 'First Transaction Time Order',
        'value_col':    'Correct First Attempt'
    }
    dm.assign_2D_sparse_ftrs(**assign_2D_sparse_ftrs_params)

    #now create the same but for views
    assign_2D_sparse_ftrs_params['ftr_name'] = 'X_views'
    assign_2D_sparse_ftrs_params['value_col'] = 'Problem View'
    dm.assign_2D_sparse_ftrs(**assign_2D_sparse_ftrs_params)

    #now create the same but for total_corrects
    assign_2D_sparse_ftrs_params['ftr_name'] = 'X_correct_cnt'
    assign_2D_sparse_ftrs_params['value_col'] = 'Corrects'
    dm.assign_2D_sparse_ftrs(**assign_2D_sparse_ftrs_params)

    #now create the same but for total_incorrects
    assign_2D_sparse_ftrs_params['ftr_name'] = 'X_incorrect_cnt'
    assign_2D_sparse_ftrs_params['value_col'] = 'Incorrects'
    dm.assign_2D_sparse_ftrs(**assign_2D_sparse_ftrs_params)

    return dm

def create_latest_ftrs(dm):
    '''Create the latest correct/view for every tm_order'''
    assign_2D_sparse_ftrs_max_params = {
        'ftr_name':     'X_correct',
        'max_ftr_name': 'X_correct_latest'
    }
    dm.assign_2D_sparse_ftrs_max(**assign_2D_sparse_ftrs_max_params)
    assign_2D_sparse_ftrs_max_params = {
        'ftr_name':     'X_views',
        'max_ftr_name': 'X_views_latest'
    }
    dm.assign_2D_sparse_ftrs_max(**assign_2D_sparse_ftrs_max_params)
    assign_2D_sparse_ftrs_max_params = {
        'ftr_name':     'X_correct_cnt',
        'max_ftr_name': 'X_correct_cnt_latest'
    }
    dm.assign_2D_sparse_ftrs_max(**assign_2D_sparse_ftrs_max_params)
    assign_2D_sparse_ftrs_max_params = {
        'ftr_name':     'X_incorrect_cnt',
        'max_ftr_name': 'X_incorrect_cnt_latest'
    }
    dm.assign_2D_sparse_ftrs_max(**assign_2D_sparse_ftrs_max_params)

    return dm

def load_preprocessed_ftrs():
    '''return a data_manip object with the preprocessed ftrs'''
    dm = lu.load_pickle('test_preprocessed_DM', 'saved_ftrs')

    load_params = {
        'ftr_name':'X_correct',
        'filenames':'X_correct',
        'foldername':'saved_ftrs'
    }
    dm.load_2D_sparse_ftrs(**load_params)
    # load_params = {
    #     'ftr_name':'X_views',
    #     'filenames':'X_views',
    #     'foldername':'saved_ftrs'
    # }
    # dm.load_2D_sparse_ftrs(**load_params)
    load_params = {
        'ftr_name':'X_correct_cnt',
        'filenames':'X_correct_cnt',
        'foldername':'saved_ftrs'
    }
    dm.load_2D_sparse_ftrs(**load_params)
    load_params = {
        'ftr_name':'X_incorrect_cnt',
        'filenames':'X_incorrect_cnt',
        'foldername':'saved_ftrs'
    }
    dm.load_2D_sparse_ftrs(**load_params)

    load_params = {
        'ftr_name':'X_correct_latest',
        'filenames':['X_correct_latest_1'],#, 'X_correct_latest_2', 'X_correct_latest_3', 'X_correct_latest_4'],
        'foldername':'saved_ftrs'
    }
    dm.load_2D_sparse_ftrs(**load_params)
    # load_params = {
    #     'ftr_name':'X_views_latest',
    #     'filenames':['X_views_latest_1', 'X_views_latest_2', 'X_views_latest_3', 'X_views_latest_4'],
    #     'foldername':'saved_ftrs'
    # }
    # dm.load_2D_sparse_ftrs(**load_params)
    load_params = {
        'ftr_name':'X_correct_cnt_latest',
        'filenames':['X_correct_cnt_latest_1'],#, 'X_correct_cnt_latest_2', 'X_correct_cnt_latest_3', 'X_correct_cnt_latest_4'],
        'foldername':'saved_ftrs'
    }
    dm.load_2D_sparse_ftrs(**load_params)
    load_params = {
        'ftr_name':'X_incorrect_cnt_latest',
        'filenames':['X_incorrect_cnt_latest_1'],#, 'X_incorrect_cnt_latest_2', 'X_incorrect_cnt_latest_3', 'X_incorrect_cnt_latest_4'],
        'foldername':'saved_ftrs'
    }
    dm.load_2D_sparse_ftrs(**load_params)

    return dm

def test_baselines():
    '''Test Baselines'''
    test_results = []

    # #Uniform_Random_Learner
    # test_2D_sparse_params = {
    #     'Learner':Learners.Uniform_Random_Learner,
    #     'tm_orders_totest':tm_orders_totest,
    #     'ftr_names':['X_correct_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{},
    #     'pred_params':{
    #         'threshold':0.5
    #     }
    # }
    # test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

    #Global_Average_Learner
    test_2D_sparse_params = {
        'Learner':Learners.Gloabl_Avg_Learner,
        'tm_orders_totest':tm_orders_totest,
        'ftr_names':['X_correct_latest'],
        'answers_name':'X_correct',
        'fit_params':{},
        'pred_params':{
            'threshold':0.5
        }
    }
    test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

    #Within_XCol_Avg_Learner (within each Problem-Step)
    test_2D_sparse_params = {
        'Learner':Learners.Within_XCol_Avg_Learner,
        'tm_orders_totest':tm_orders_totest,
        'ftr_names':['X_correct_latest'],
        'answers_name':'X_correct',
        'fit_params':{},
        'pred_params':{
            'threshold':0.5
        }
    }
    test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

    #Within_XRow_Avg_Learner (within each Student)
    test_2D_sparse_params = {
        'Learner':Learners.Within_XRow_Avg_Learner,
        'tm_orders_totest':tm_orders_totest,
        'ftr_names':['X_correct_latest'],
        'answers_name':'X_correct',
        'fit_params':{},
        'pred_params':{
            'threshold':0.5
        }
    }
    test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

    return test_results

def custdist_test(test_2D_sparse_params, w_incor, agg, thresh):
    info()

    test_2D_sparse_params['fit_params']['w_incorrect'] = w_incor
    test_2D_sparse_params['fit_params']['agg'] = agg
    test_2D_sparse_params['pred_params']['threshold'] = thresh
    test_results = dm.test_2D_sparse(**test_2D_sparse_params)

    w_incor = str(w_incor).replace('.', 'pt')
    agg = str(agg).replace('.', 'pt')
    thresh = str(thresh).replace('.', 'pt')

    lu.save_pickle('custdist_20tm_wincor{}_agg{}_th{}'.format(w_incor, agg, thresh),\
        test_results, 'results_parallel')

    return test_results

def info():
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    # pool = mp.Pool(processes=6)
    dm = load_preprocessed_ftrs()
    test_results = []
    tm_orders_totest = 20
    # test_results.append(test_baselines())

    # '''3.) Test NN methods'''
    # #NN_cosine with 0/1 encoding in X
    # test_2D_sparse_params = {
    #     'Learner':Learners.NN_cos_Learner,
    #     'tm_orders_totest':tm_orders_totest,
    #     'ftr_names':['X_correct_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{},
    #     'pred_params':{
    #         'threshold':0.5
    #     }
    # }
    # test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))
    #
    #NN cosine with 0/1 encoding in X but simdiags = 0
    # test_2D_sparse_params = {
    #     'Learner':Learners.NN_cos_noselfsim_Learner,
    #     'tm_orders_totest':tm_orders_totest,
    #     'ftr_names':['X_correct_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{},
    #     'pred_params':{
    #         'threshold':0.5
    #     }
    # }
    # test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))
    #
    # #NN_cosine with -1/0/1 encoding in X (-1 for incorrects instead of 0)
    # test_2D_sparse_params = {
    #     'Learner':Learners.NN_cos_encode_wrong_Learner,
    #     'tm_orders_totest':tm_orders_totest,
    #     'ftr_names':['X_correct_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{
    #         'encode_wrong':-1.0
    #     },
    #     'pred_params':{
    #         'threshold':0.0
    #     }
    # }
    # test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))
    #
    # #NN_l1 with 0/1 encoding in X
    # test_2D_sparse_params = {
    #     'Learner':Learners.NN_l1_Learner,
    #     'tm_orders_totest':tm_orders_totest,
    #     'ftr_names':['X_correct_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{},
    #     'pred_params':{
    #         'threshold':0.5
    #     }
    # }
    # test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))
    #
    # #NN_l1 with -1/0/1 encoding in X
    # test_2D_sparse_params = {
    #     'Learner':Learners.NN_l1_encode_wrong_Learner,
    #     'tm_orders_totest':tm_orders_totest,
    #     'ftr_names':['X_correct_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{
    #         'encode_wrong':-1.0
    #     },
    #     'pred_params':{
    #         'threshold':0.0
    #     }
    # }
    # test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

    # # NN with custom encoding using correct_cnt and incorrect_cnt
    # test_2D_sparse_params = {
    #     'Learner':Learners.NN_custom_withweights_Learner,
    #     'tm_orders_totest':20,
    #     'ftr_names':['X_correct_latest', 'X_correct_cnt_latest', 'X_incorrect_cnt_latest'],
    #     'answers_name':'X_correct',
    #     'fit_params':{
    #         'w_correct':1.0,
    #         'w_incorrect':1.0,
    #         'agg':'sum',
    #         'load':False,
    #         'save':False
    #     },
    #     'pred_params':{
    #         'threshold':0.3
    #     },
    #     'ftr_name':'X_cor_and_incor_cnt'
    # }

    #different hyperparams
    import ipdb; ipdb.set_trace()

    # NN with custom encoding using correct_cnt and incorrect_cnt
    test_2D_sparse_params = {
        'Learner':Learners.NN_custom_withweights_Learner,
        'tm_orders_totest':20,
        'ftr_names':['X_correct_latest', 'X_correct_cnt_latest', 'X_incorrect_cnt_latest'],
        'answers_name':'X_correct',
        'fit_params':{
            'w_correct':1.0,
            'w_incorrect':10.0,
            'agg':'sqrt(sum_of_squares)',
            'load':False,
            'save':False
        },
        'pred_params':{
            'threshold':0.5
        },
        'ftr_name':'X_cor_and_incor_cnt'
    }

    dm.test_2D_sparse(**test_2D_sparse_params)

    # thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]
    # for thresh in thresholds:
    #     test_2D_sparse_params['pred_params']['threshold'] = thresh
    #     test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))
    #
    # lu.save_pickle('varthresh_rowavg', test_results, 'hyperparam_tuning_results')

    import ipdb; ipdb.set_trace()
    '''Run multiple tests in parralell CANT BE RUN WITH IPDB'''
    # w_incors = [0.1, 1.0, 10.0]
    # aggs = ['sum', 'avg', 'sqrt(sum_of_squares)']
    # thresholds = [0.3, 0.5, 0.7]
    #
    # combinations = [(test_2D_sparse_params, w_incor, agg, thresh) \
    # for w_incor in w_incors for agg in aggs for thresh in thresholds]
    #
    # pool = mp.Pool(processes=3)
    # results = pool.starmap(custdist_test, combinations)
    #
    # lu.save_pickle('custdost_25tm_overall_results', results, 'results')

    '''4.) calculate/graph results'''
    plot_test_2D_sparse_results_params = {
        'results':test_results,
        'x_range_col':'t_cur',
        'label_col':'Learner Name',
        'value_cols':['Accuracy', 'Positive Predictive Value', 'Negative Predictive Value',\
            'True Positive Rate', 'True Negative Rate'],
        'starting_figure':0
    }
    gu.plot_test_2D_sparse_results(**plot_test_2D_sparse_results_params)

    '''plot in a 3x3 grid the test results of cust_dist'''
    # aggs = ['sum', 'avg', 'sqrt(sum_of_squares)']
    # wincors = ['10pt0', '1pt0', '0pt1']
    # x_range = np.arange(20)
    # y_ticks = 0.70 + np.arange(31)*0.01
    #
    # f, axarr = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    # for i, agg in enumerate(aggs):
    #     for j, wincor in enumerate(wincors):
    #         #load the 3 results of diff wincors
    #         th0pt3 = lu.load_pickle('custdist_20tm_wincor{}_agg{}_th0pt3'.format(wincor, agg), 'results_parallel')
    #         th0pt5 = lu.load_pickle('custdist_20tm_wincor{}_agg{}_th0pt5'.format(wincor, agg), 'results_parallel')
    #         th0pt7 = lu.load_pickle('custdist_20tm_wincor{}_agg{}_th0pt7'.format(wincor, agg), 'results_parallel')
    #
    #         #plot in subplot i,j
    #         axarr[i, j].plot(x_range, th0pt3.loc[:, 'Accuracy'], label='wincor={}|agg={}|thresh=0pt3'.format(wincor, agg))
    #         axarr[i, j].plot(x_range, th0pt5.loc[:, 'Accuracy'], label='wincor={}|agg={}|thresh=0pt5'.format(wincor, agg))
    #         axarr[i, j].plot(x_range, th0pt7.loc[:, 'Accuracy'], label='wincor={}|agg={}|thresh=0pt7'.format(wincor, agg))
    #
    #         axarr[i, j].legend(loc='best', prop={'size':5})
    #         axarr[i, j].set_xlabel('tm')
    #         axarr[i, j].set_ylabel('acc')
    #         axarr[i, j].set_xticks(x_range)
    #         axarr[i, j].set_yticks(y_ticks)
    #         axarr[i, j].grid()
    #
    # plt.show()
