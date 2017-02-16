
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

def plot_test_2D_sparse_results(results, x_range_col, label_col, value_cols, starting_figure):
    '''
    Inputs:
        [results] = list of dataframes that contain the results
        label_col = name of the col seperate difference methods
        [value_cols] = list of different valued graphs to plot
    '''
    #get the xrange from an arbitrary result (all should be same)
    x_range = results[0][x_range_col]

    for i, value in enumerate(value_cols):
        plt.figure(i+starting_figure)
        for result in results:
            y_range = result[value]
            plt.plot(x_range, y_range, label=result[label_col][0])

        plt.legend(loc='best', prop={'size':9})
        plt.xlabel(x_range_col)
        plt.ylabel(value)
        plt.xticks(x_range)
        plt.yticks(0+np.arange(26)*0.04) #% from 70% to 100%
        plt.title('{} by {}'.format(value, x_range_col))
        plt.grid()
        plt.show()

    print('graphing complete')

def to_percent(x, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * x)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def pd_dseries_hist(dseries, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis):
    '''takes a pandas DataSeries and plots the coressponding histogram'''
    dseries.hist(bins=math.floor(dseries.size*prct_cnt_bins))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if prct_x_axis:
        formatter = FuncFormatter(to_percent)
        plt.gca().xaxis.set_major_formatter(formatter)

    print('plotting', title)
    plt.show()
    print('plot closed')

def plot_df_stacked(df, kind, xlabel, ylabel, title):
    '''take dataframe and plot it as stacked prct area with labels and titles'''
    ax = df.plot(kind=kind, stacked=True)
    # make y-axis a prct
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    print('plotting', title)
    plt.show()
    print('plot closed')

def plot_df(df, kind, xlabel, ylabel, title):
    '''take dataframe and plot it as a line prct with labels and titles'''
    ax = df.plot(kind=kind)
    # make y-axis a prct
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    print('plotting', title)
    plt.show()
    print('plot closed')

def plot_test(results, Evaluator, xlabel, ylabel, title):
    '''
    Assuming results is a list of columns that result from DataManipulator.test_predictions_cum_tmrank
    Evaluator is a string for which metric you are looking at
    '''
    results = pd.DataFrame(results)
    results.drop('tm_rank_pred_range', axis=1, inplace=True)
    results.set_index(['tm_rank_train_range', 'Evaluator', 'Learner'], inplace=True)
    results = results.unstack(['Evaluator', 'Learner'])
    results.index = results.index.map(lambda x: int(x[x.index(',')+1:x.index(']')]))
    results.sort_index(inplace=True)
    results['eval_value', Evaluator].plot(title = title, grid=True)
    plt.legend(loc='best', ncol=1, fontsize=6.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    print('plotting', title)
    plt.show()
    print('plot closed')
