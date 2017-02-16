import DataManipulators
import Learners
import loading_util as lu
import graphing_util as gu

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
'''
this script is meant to show the results of testing on single KC data
'''

# '''Prelim A: Load single KC data'''
# single_KC_data = pd.read_pickle('/Users/qipanda/Documents/2016-2017_KDD_Thesis/education_data'+\
#     '/bridge_to_algebra_2008_2009/bridge_0809_KC.pkl')
#
# test_df = pd.DataFrame(
#     {
#         'Anon Student Id':['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
#         'Problem Name':['P1', 'P1', 'P2', 'P2', 'P1', 'P1', 'P2', 'P2', 'P1', 'P1', 'P1', 'P1', 'P2'],
#         'Step Name':['S1', 'S1', 'S1', 'S2', 'S2', 'S1', 'S3', 'S3', 'S1', 'S1', 'S1', 'S1', 'S3'],
#         # 'First Transaction Time Rank':[1, 1, 2, 1, 2, 1, 2, 1, 2, 3],
#         'First Transaction Time':['2008-09-01', '2008-09-02', '2008-09-03', '2008-09-04',\
#             '2008-09-05', '2008-09-01', '2008-09-02', '2008-09-03', '2008-09-01', '2008-09-02',\
#             '2008-09-03', '2008-09-04', '2008-09-05'],
#         'Correct First Attempt':[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
#         'Problem View':[1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 3, 4, 1]
#     }
# )
#
# dm = DataManipulators.DataManipulator(df=single_KC_data)
#
# '''Prelim B: assign tm_ranks to the data'''
# assign_rel_tm_params = {
#     'granularity_grp':  ['Anon Student Id', 'Problem Name', 'Problem View'],
#     'time_within_col':  'Anon Student Id',
#     'time_col':         'First Transaction Time',
#     'tm_order_col':     'First Transaction Time Order'
# }
# dm.assign_rel_tm(**assign_rel_tm_params)

# '''1.) Preprocess features in DM'''
# assign_2D_sparse_ftrs_params = {
#     'ftr_name':     'X_correct',
#     'sim_col':      'Anon Student Id',
#     'ftr_cols':     ['Problem Name', 'Step Name'],
#     'tm_order_col': 'First Transaction Time Order',
#     'value_col':    'Correct First Attempt'
# }
# dm.assign_2D_sparse_ftrs(**assign_2D_sparse_ftrs_params)
#
# #now create the same but for views
# assign_2D_sparse_ftrs_params['ftr_name'] = 'X_views'
# assign_2D_sparse_ftrs_params['value_col'] = 'Problem View'
# dm.assign_2D_sparse_ftrs(**assign_2D_sparse_ftrs_params)
#
# #now create the latest correct/view for every tm_order
# assign_2D_sparse_ftrs_max_params = {
#     'ftr_name':     'X_correct',
#     'max_ftr_name': 'X_correct_latest'
# }
# dm.assign_2D_sparse_ftrs_max(**assign_2D_sparse_ftrs_max_params)
# assign_2D_sparse_ftrs_max_params = {
#     'ftr_name':     'X_views',
#     'max_ftr_name': 'X_views_latest'
# }
# dm.assign_2D_sparse_ftrs_max(**assign_2D_sparse_ftrs_max_params)
#
#save the preprocessed dm
# lu.save_pickle('test_preprocessed_DM', dm, 'saved_ftrs')

'''1.) Preprocessing for based X_ftrs, X_views done, load them'''
dm = lu.load_pickle('test_preprocessed_DM', 'saved_ftrs')
tm_orders_totest = 50

load_params = {
    'ftr_name':'X_correct',
    'filenames':'X_correct',
    'foldername':'saved_ftrs'
}
dm.load_2D_sparse_ftrs(**load_params)
load_params = {
    'ftr_name':'X_views',
    'filenames':'X_views',
    'foldername':'saved_ftrs'
}
dm.load_2D_sparse_ftrs(**load_params)
load_params = {
    'ftr_name':'X_correct_latest',
    'filenames':['X_correct_latest_1', 'X_correct_latest_2', 'X_correct_latest_3', 'X_correct_latest_4'],
    'foldername':'saved_ftrs'
}
dm.load_2D_sparse_ftrs(**load_params)
load_params = {
    'ftr_name':'X_views_latest',
    'filenames':['X_views_latest_1', 'X_views_latest_2', 'X_views_latest_3', 'X_views_latest_4'],
    'foldername':'saved_ftrs'
}
dm.load_2D_sparse_ftrs(**load_params)

test_results = []
'''2.) Test Baselines'''
#Uniform_Random_Learner
# test_2D_sparse_params = {
#     'Learner':Learners.Uniform_Random_Learner,
#     'tm_orders_totest':tm_orders_totest,
#     'ftr_name':'X_correct_latest',
#     'answers_name':'X_correct',
#     'fit_params':{},
#     'pred_params':{
#         'threshold':0.5
#     }
# }
# nn_uniform_results = dm.test_2D_sparse(**test_2D_sparse_params)

#Global_Average_Learner
test_2D_sparse_params = {
    'Learner':Learners.Gloabl_Avg_Learner,
    'tm_orders_totest':tm_orders_totest,
    'ftr_name':'X_correct_latest',
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
    'ftr_name':'X_correct_latest',
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
    'ftr_name':'X_correct_latest',
    'answers_name':'X_correct',
    'fit_params':{},
    'pred_params':{
        'threshold':0.5
    }
}
test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

'''3.) Test NN methods'''
#NN_cosine with 0/1 encoding in X
test_2D_sparse_params = {
    'Learner':Learners.NN_cos_Learner,
    'tm_orders_totest':tm_orders_totest,
    'ftr_name':'X_correct_latest',
    'answers_name':'X_correct',
    'fit_params':{},
    'pred_params':{
        'threshold':0.5
    }
}
test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

#NN cosine with 0/1 encoding in X but simdiags = 0
test_2D_sparse_params = {
    'Learner':Learners.NN_cos_noselfsim_Learner,
    'tm_orders_totest':tm_orders_totest,
    'ftr_name':'X_correct_latest',
    'answers_name':'X_correct',
    'fit_params':{},
    'pred_params':{
        'threshold':0.5
    }
}
test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))

#NN_cosine with -1/0/1 encoding in X (-1 for incorrects instead of 0)
test_2D_sparse_params = {
    'Learner':Learners.NN_cos_encode_wrong_Learner,
    'tm_orders_totest':tm_orders_totest,
    'ftr_name':'X_correct_latest',
    'answers_name':'X_correct',
    'fit_params':{
        'encode_wrong':-1.0
    },
    'pred_params':{
        'threshold':0.0
    }
}
test_results.append(dm.test_2D_sparse(**test_2D_sparse_params))
# import ipdb; ipdb.set_trace()

'''#TODO calculate/graph results'''
plot_test_2D_sparse_results_params = {
    'results':test_results,
    'x_range_col':'t_cur',
    'label_col':'Learner Name',
    'value_cols':['Accuracy', 'Positive Predictive Value', 'Negative Predictive Value',\
        'True Positive Rate', 'True Negative Rate'],
    'starting_figure':0
}
gu.plot_test_2D_sparse_results(**plot_test_2D_sparse_results_params)
lu.save_pickle('test_results_3methods_50tm', test_results, 'results')
