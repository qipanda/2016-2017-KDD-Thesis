import pandas as pd
import numpy as np
import math
import pickle
import os.path

def save_pickle_workaround(filename, value, foldername=None):
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(value)

    if foldername is None:
        print('Saving pickle to {}.pickle'.format(filename))
        with open('{}.pickle'.format(filename), 'wb') as handle:
            for idx in range(0, n_bytes, max_bytes):
                handle.write(bytes_out[idx:idx+max_bytes])
    else:
        print('Saving pickle to {}/{}.pickle'.format(foldername, filename))
        with open('{}/{}.pickle'.format(foldername, filename), 'wb') as handle:
            for idx in range(0, n_bytes, max_bytes):
                handle.write(bytes_out[idx:idx+max_bytes])

def load_pickle_workaround(filename, foldername=None):
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)

    if foldername is None:
        print('Loading pickle from {}.pickle'.format(filename))
        input_size = os.path.getsize('{}.pickle'.format(filename))

        with open('{}.pickle'.format(filename), 'rb') as handle:
            for _ in range(0, input_size, max_bytes):
                bytes_in += handle.read(max_bytes)
    else:
        print('Loading pickle from {}/{}.pickle'.format(foldername, filename))
        input_size = os.path.getsize('{}/{}.pickle'.format(foldername, filename))

        with open('{}/{}.pickle'.format(foldername, filename), 'rb') as handle:
            for _ in range(0, input_size, max_bytes):
                bytes_in += handle.read(max_bytes)
    return pickle.loads(bytes_in)

def save_pickle(filename, value, foldername=None):
    if foldername is None:
        print('Saving pickle to {}.pickle'.format(filename))
        with open('{}.pickle'.format(filename), 'wb') as handle:
            pickle.dump(value, handle)
    else:
        print('Saving pickle to {}/{}.pickle'.format(foldername, filename))
        with open('{}/{}.pickle'.format(foldername, filename), 'wb') as handle:
            pickle.dump(value, handle)

def load_pickle(filename, foldername=None):
    if foldername is None:
        print('Loading pickle from {}.pickle'.format(filename))
        with open('{}.pickle'.format(filename), 'rb') as handle:
            return pickle.load(handle)
    else:
        print('Loading pickle from {}/{}.pickle'.format(foldername, filename))
        with open('{}/{}.pickle'.format(foldername, filename), 'rb') as handle:
            return pickle.load(handle)

def load_data(src_path, allrows, numrows=1):
    if allrows:
        return pd.read_csv(filepath_or_buffer=src_path, sep='\t')
    else:
        return pd.read_csv(filepath_or_buffer=src_path, sep='\t', nrows=numrows)

def load_pickle_full_df(src_folder):
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
