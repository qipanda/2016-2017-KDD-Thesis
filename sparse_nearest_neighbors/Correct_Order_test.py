import DataManipulators
import Learners
import loading_util as lu
import graphing_util as gu

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import ipdb; ipdb.set_trace()

single_KC_data = pd.read_pickle('/Users/qipanda/Documents/2016-2017_KDD_Thesis/education_data'+\
    '/bridge_to_algebra_2008_2009/bridge_0809_KC.pkl')

df = single_KC_data.loc[:, ['Anon Student Id', 'Problem Name', 'Step Name', 'Problem View', 'Correct First Attempt']]
ndarray = df.set_index(['Anon Student Id', 'Problem Name', 'Step Name', 'Problem View']).unstack().values

for j in range(ndarray.shape[1] - 1):
    #for each column after the first, check if >= last column (problem view)
    non_null_idx = ~np.isnan(ndarray[:, j+1]) & ~np.isnan(ndarray[:, j])
    np.all(ndarray[non_null_idx, j+1] >= ndarray[non_null_idx, j])
