
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

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

def plot_df_stacked(df, xlabel, ylabel, title):
    '''take dataframe and plot it as stacked prct area with labels and titles'''
    ax = df.plot(kind='area', stacked=True)
    # make y-axis a prct
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    print('plotting', title)
    plt.show()
    print('plot closed')

def plot_df_line(df, xlabel, ylabel, title):
    '''take dataframe and plot it as a line prct with labels and titles'''
    ax = df.plot(kind='line')
    # make y-axis a prct
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    print('plotting', title)
    plt.show()
    print('plot closed')
