import DataManipulators
import Learners
import Evaluators
import loading_util as lu
import graphing_util as gu

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
'''
this script is meant to show the results of testing on single KC data
'''

'''Prelim A: Load single KC data'''
single_KC_data = pd.read_pickle('/Users/qipanda/Documents/2016-2017_KDD_Thesis/education_data'+\
    '/bridge_to_algebra_2008_2009/bridge_0809_KC.pkl')

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

dm = DataManipulators.DataManipulator(df=single_KC_data)

'''Prelim B: assign tm_ranks to the data'''
assign_rel_tm_params = {
    'granularity_grp':  ['Anon Student Id', 'Problem Name', 'Problem View'],
    'time_within_col':  'Anon Student Id',
    'time_col':         'First Transaction Time',
    'tm_order_col':     'First Transaction Time Order'
}
dm.assign_rel_tm(**assign_rel_tm_params)

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
# dm = lu.load_pickle('test_preprocessed_DM', 'saved_ftrs')
import ipdb; ipdb.set_trace()
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


'''2.) Test for cosine_sim, 0/1 encoding'''
test_2D_sparse_params = {
    'Learner':Learners.NN_cos_Learner,
    'tm_orders_totest':20,
    'ftr_name':'X_correct_latest',
    'answers_name':'X_correct',
    'fit_params':{},
    'pred_params':{
        'threshold':0.5
    }
}
nn_01_results = dm.test_2D_sparse(**test_2D_sparse_params)
print('hi')
#TODO calculate/graph results

# '''Preprocess data for NN'''
# preprocess_params = {
#     'sim_col':'Anon Student Id',
#     'grp_key':['Problem Name', 'Step Name'],
#     'tm_col':'First Transaction Time Rank',
#     'attempt_col':'Problem View',
#     'pred_col':'Correct First Attempt'
# }
# predict_params = {
#     't_cur':0,
#     'thresh':0.0
# }
# nn = Learners.NN_Learner(dm.df)
# nn.preprocess(**preprocess_params)
#
# '''Testing across tm_ranges and getting True Pos, False Pos, True Neg, False Neg'''
# import ipdb; ipdb.set_trace()
#
# results = {'t_cur':[], 'TP':[], 'FP':[], 'TN':[], 'FN':[], 'Total':[]}
# t_range = range(150)
# for t_cur in t_range:
#     print(t_cur)
#     predict_params['t_cur'] = t_cur
#     preds, actls = nn.predict(**predict_params)
#     results['t_cur'].append(t_cur)
#     results['TP'].append(np.sum(preds[preds==True] == actls[preds==True]))
#     results['FP'].append(np.sum(preds[preds==True] != actls[preds==True]))
#     results['TN'].append(np.sum(preds[preds==False] == actls[preds==False]))
#     results['FN'].append(np.sum(preds[preds==False] != actls[preds==False]))
#     results['Total'].append(preds.shape[0])
# import ipdb; ipdb.set_trace()
# # lu.save_pickle(filename='NN_sensitivity_150', value=results)
# results = lu.load_pickle(filename='NN_sensitivity_150')
#
# results_df = pd.DataFrame(results)
# results_df['Accuracy'] = (results_df['TP'] + results_df['TN'])/results_df['Total']
# results_df['Precision'] = results_df['TP']/(results_df['TP'] + results_df['FP'])
# results_df['Recall'] = results_df['TP']/(results_df['TP'] + results_df['FN'])
#
# results_df['TP_prct'] = results_df['TP']/results_df['Total']
# results_df['FP_prct'] = results_df['FP']/results_df['Total']
# results_df['TN_prct'] = results_df['TN']/results_df['Total']
# results_df['FN_prct'] = results_df['FN']/results_df['Total']
#
# plt.plot(range(150), results_df['TP_prct'], color='g', label='TP_prct')
# plt.plot(range(150), results_df['FP_prct'], color='r', label='FP_prct')
# plt.plot(range(150), results_df['TN_prct'], color='g', label='TN_prct', ls='--')
# plt.plot(range(150), results_df['FN_prct'], color='r', label='FN_prct', ls='--')
# plt.plot(range(150), results_df['Accuracy'], color='b', label='Acc.')
# plt.legend(loc='best')
# plt.xlabel('Time')
# plt.ylabel('Percentage (%)')
# plt.grid()
# plt.show()
