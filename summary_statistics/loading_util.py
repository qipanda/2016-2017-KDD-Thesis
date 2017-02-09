import pandas as pd
import numpy as np
import math

def load_data(src_path, allrows, numrows=1):
    if allrows:
        return pd.read_csv(filepath_or_buffer=src_path, sep='\t')
    else:
        return pd.read_csv(filepath_or_buffer=src_path, sep='\t', nrows=numrows)

def load_pickle(src_folder):
    raw_data1 = pd.read_pickle(src_folder + '/bridge_0809_raw1.pkl')
    raw_data2 = pd.read_pickle(src_folder + '/bridge_0809_raw2.pkl')
    raw_data = raw_data1.append(raw_data2)

    return raw_data

def create_single_KC_data(KC, KC_col_name):
    raw_data = load_data('/Users/qipanda/Documents/2017_Thesis/education_data'+\
        '/bridge_to_algebra_2008_2009/bridge_to_algebra_2008_2009_train.txt', True)

    KC_problem_list = raw_data.loc[raw_data[KC_col_name].str.contains(KC, na=False)]
    KC_problem_list = pd.DataFrame(KC_problem_list['Problem Name'].drop_duplicates())

    single_KC_data = pd.merge(raw_data, KC_problem_list, on='Problem Name')

    single_KC_data.to_pickle('/Users/qipanda/Documents/2017_Thesis/education_data'+\
        '/bridge_to_algebra_2008_2009/bridge_0809_' + KC + '.pkl')

if __name__ == '__main__':
    '''if this script is called it means we want to save some pickle file'''
    create_single_KC_data('Calculate sum digit -- no carry-1', 'KC(KTracedSkills)')

    #Load the data
    # raw_data = load_data('/Users/qipanda/Documents/2017_Thesis/education_data'+\
    #     '/bridge_to_algebra_2008_2009/bridge_to_algebra_2008_2009_train.txt', True)

    # #need to split data into two to avoid Issue 24658 NOT RESOLVED AS OF DEC 18, 2016
    # half = math.floor(raw_data.shape[0]/2)
    # raw_data1 = raw_data.ix[0:(half-1)]
    # raw_data2 = raw_data.ix[half:]
    #
    # raw_data1.to_pickle('/Users/qipanda/Documents/2017_Thesis/education_data'+\
    #     '/bridge_to_algebra_2008_2009/bridge_0809_raw1.pkl')
    # raw_data2.to_pickle('/Users/qipanda/Documents/2017_Thesis/education_data'+\
    #     '/bridge_to_algebra_2008_2009/bridge_0809_raw2.pkl')
