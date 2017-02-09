import pandas as pd
import numpy as np

import graphing_util as gu
import loading_util as lu

class SummaryStats(object):
    '''a class that holds imported pandas dataframe and can perform various
    summary statistics on it using groupbys and aggregation functions'''

    def __init__(self, df=pd.DataFrame()):
        self.df = df

    # @classmethod
    # def with_df(cls, df):
    #     return cls(df)

    def grpby_cntdist_prct(self, grpby_terms, count_cols):
        '''
        For internal dataframe, count unique [count_cols] grouped by [grpby_terms]
        Inputs:
            grpby_terms = list of strings between [1, inf]
            count_cols = list of strings between [1, inf]
        Algorithm:
        1.) Filter self.df to only grpby_terms and count_cols
        2.) Remove all duplicate rows (this in effect does the dist part of the cnt)
        3.) Take note of the number of unique count_cols
        3.) Create GroupBy object using grpby_terms
        4.) Perform count of rows for each group and divide by number of unique
            count_cols to see it as a prct of total count_cols
        Outputs:
            results = the resulting Pandas Series
        '''
        results = self.df[grpby_terms + count_cols]
        results = results.drop_duplicates()

        unique_count_cols = len(results.groupby(by=count_cols)) #number of unique count_cols
        results = results.groupby(by=grpby_terms)
        results = results.size()/unique_count_cols #this is a pandas series
        print('GroupBy:', grpby_terms, ' uniquely counting:', count_cols, 'complete')

        return results

    def grpby_avg(self, grpby_terms, avg_col):
        '''
        For internal dataframe, avg [avg_col] grouped by [grpby_terms]
        Inputs:
            grpby_terms = list of strings between size [1, inf]
            avg_cols = string of col name to be averaged (given as a list of 1 str)
        Algorithm:
        1.) Filter self.df to only relevant terms ([grpby_terms and avg_col])
        2.) Group by [grpby_terms]
        3.) Take average of [avg_col]
        4.) Return agg_results Pandas Series of results
        Outputs:
            results = the resulting Pandas Series
        '''
        results = self.df[grpby_terms + avg_col]
        results = results.groupby(by=grpby_terms)
        results = results.mean()
        print('GroupBy:', grpby_terms, ' averaging:', avg_col, 'complete')

        return results

    def split_stack_cntdist_prct(self, split_col, delim, count_cols):
        '''
        Inputs:
            split_col = String of the col to be split by, also the groupby
            delim = delimiter to be split by
            count_cols = list of strings between [1, inf]
        split slef.df by split_col, stack rows by what it split
        Algorithm:
            1.) select only relevent rows [split_col and count_cols]
                remove dups and remove rows where split_col is null
            2.) take [split_col] as a Series and split by delim, make this into
                a nested list (.tolist()) and put into dataframe
            3.) merge this with the [count_cols] into a big dataframe (joins
                on new reset index)
            4.) drop the "variable" column and all null values (None in Pandas),
                and any remaining duplicates of (stud-single_KC) afterwards
            5.) count the number of unique [count_cols]
            6.) groupby and divide for %
        '''

        results = self.df[[split_col] + count_cols].drop_duplicates()
        results = results.loc[results[split_col].notnull()].reset_index()
        split_cols = pd.DataFrame(results[split_col].str.split(delim).tolist())

        results = split_cols.join(results[count_cols])
        results = pd.melt(results, id_vars=count_cols, \
            value_vars=[x for x in range(0, results.shape[1]-len(count_cols))], \
            value_name=split_col)
        results = results.drop('variable', 1).dropna().drop_duplicates()

        unique_count_cols = self.df[count_cols[0]].nunique() #number of unique count_cols
        results = results.groupby(split_col).size()/unique_count_cols

        return results


if __name__ == '__main__':
    '''if this script is called as the main script, run a summary of the data and plot them'''

    #load data from pickle
    raw_data = lu.load_pickle('/Users/qipanda/Documents/2017_Thesis/education_data'+\
        '/bridge_to_algebra_2008_2009')

    #create SummaryStats object with Dataframe
    SS = SummaryStats(raw_data)

    #Plot 1.1, hist: number of students that have attempted x unique problems
    title = 'Number of Students that have Attempted x prct. Unique Problems'
    xlabel = 'Prct. of Unique Problems'
    ylabel = 'Freq. of Students'
    prct_cnt_bins = 0.10
    prct_x_axis = True
    cnt_result = SS.grpby_cntdist_prct(['Anon Student Id'], ['Problem Name'])

    gu.pd_dseries_hist(cnt_result, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis)

    #Plot 1.2, hist: number of students that have attempted x unique steps
    #this is given by problem-step combinations
    title = 'Number of Students that have Attempted x prct. Unique Steps'
    xlabel = 'Prct. of Unique Steps'
    ylabel = 'Freq. of Students'
    prct_cnt_bins = 0.10
    prct_x_axis = True
    agg_dseries = SS.grpby_cntdist_prct(['Anon Student Id'], ['Problem Name', 'Step Name'])

    gu.pd_dseries_hist(agg_dseries, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis)

    #Plot 2.1, hist: number of students that have x accuracy on all steps attempted
    agg_dseries = SS.grpby_avg(['Anon Student Id'], ['Correct First Attempt'])
    title = 'Number of Students that have x prct. Average Accuracy per Step'
    xlabel = 'Prct. Correct First Attempt'
    ylabel = 'Freq. of Students'
    prct_x_axis = True
    prct_cnt_bins = (1.0/agg_dseries.shape[0])*100 #want 100 bins for this one

    gu.pd_dseries_hist(agg_dseries, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis)

    #Plot 3.1, hist: number of problems that have been attempted by x prct. of students
    title = 'Number of Problems that have been Attempted by x prct. of Students'
    xlabel = 'Prct. of All Students'
    ylabel = 'Freq. of Problems'
    prct_cnt_bins = 0.001
    prct_x_axis = True
    cnt_result = SS.grpby_cntdist_prct(['Problem Name'], ['Anon Student Id'])

    gu.pd_dseries_hist(cnt_result, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis)

    #Plot 3.2, hist: number of steps that have been attempted by x prct. of students
    title = 'Number of Steps that have been Attempted by x prct. of Students'
    xlabel = 'Prct. of All Students'
    ylabel = 'Freq. of Steps'
    prct_cnt_bins = 0.0001
    prct_x_axis = True
    cnt_result = SS.grpby_cntdist_prct(['Problem Name', 'Step Name'], ['Anon Student Id'])

    gu.pd_dseries_hist(cnt_result, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis)

    #Plot 4.1, hist: number of KC's that have been attempted by x prct. of students
    title = 'Number of KCs that have been Attempted by x prct. of Students'
    xlabel = 'Prct. of All Students'
    ylabel = 'Freq. of KC\'s'
    prct_cnt_bins = 0.05
    prct_x_axis = True
    cnt_result = SS.split_stack_cntdist_prct('KC(KTracedSkills)', '~~', ['Anon Student Id'])

    gu.pd_dseries_hist(cnt_result, xlabel, ylabel, title, prct_cnt_bins, prct_x_axis)
