import numpy as np
import pandas as pd
import time
import pickle
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

import loading_util as lu

class DataManipulator(object):
    '''
    The main class deals performing prepatory datamanipulation as well as cross
    -validation splitting and the actual cross-validation testing in general
    given a methods.
    '''
    def __init__(self, df=None, sparse_ftrs=None):
        if df is None:
            df = pd.DataFrame()
        if sparse_ftrs is None:
            sparse_ftrs = {}
        self.df = df
        self.sparse_ftrs = sparse_ftrs

    def assign_rel_tm(self, granularity_grp, time_within_col, time_col, tm_order_col):
        '''
        Change [time_col] to be integer ascending according to temporal time order
        within each [time_within_grp] based on the key defined by [granularity_grp]
        Inputs:
            granularity_grp - the list of columns that define the unique key to
                grant a time rank
            time_within_col - the column to calculate relative time rank within,
                should be a sub_column of [granularity_grp]
            time_col - the column that contains the timestamp
            tm_order_col - name of the new time ranking column
        Algorithm:
            1.) Convert [time_col] to a timestamp (even if already)
            2.) Group by [granularity_grp] and take the min of [time_col] as agg col
            3.) Rank within [time_within_col] on the group by object
        Outputs:
            self.df is modified to have its [time_col] changed to relative temporal
            order within [time_within_grp]
        '''
        self.df[time_col] = pd.to_datetime(self.df[time_col])

        #convert back to dataframe and turn the series tuple label into two df cols
        time_per_gran = pd.DataFrame(self.df.groupby(granularity_grp)[time_col].min()).reset_index()

        #calculate the ranks within each [time_within_col] based on min time above
        ranks = pd.DataFrame(time_per_gran.groupby(time_within_col)[time_col].rank(ascending=True))
        ranks.columns = [tm_order_col]

        #don't need this col anymore, replaced with rank
        time_per_gran.drop(time_col, axis=1, inplace=True)

        #inner join the two by index (order was preserved for ranks since 1 column grouping)
        grps_with_ranks = pd.concat([time_per_gran, ranks], axis=1)

        #finally, inner join with the main df to add the final ranks col
        self.df = pd.merge(self.df, grps_with_ranks, on=granularity_grp)

        #remove all non-integer ranked questions (tied ones)
        self.df = self.df.loc[self.df[tm_order_col]%1==0]

    def assign_2D_sparse_ftrs(self, ftr_name, sim_col, ftr_cols, tm_order_col, value_col):
        '''
        From self.df, add to self.sprase_ftrs list of 2D sparse ftr fectors
        Inputs:
            ftr_name: name of list to add to self.sparse_ftrs dictionary
            sim_col: the name of the "rows" column in self.df
            [ftr_cols]: the name(s) of the "cols" column in self.df
            time_order_col: the name of the tm_order_col
            value_col: the name of the values in the sparse_ftrs in self.df
        Outputs:
            appended a list of sparse_ftrs to self.sparse_ftrs where each list
            index refers to a tm_order
        '''
        #filter down the df to what is needed and also combine grp_key into 1 column
        df = self.df.loc[:, [sim_col, *ftr_cols, tm_order_col, value_col]]
        df['Group Key'] = df.loc[:, [*ftr_cols]].sum(axis=1)

        #get unique counts for sparse array dimensions
        m = df[sim_col].nunique()
        n = df['Group Key'].nunique()

        #filter down to relevant columns and set index for pred_col values and attempt_col values
        df_indexed = df.loc[:, [sim_col, 'Group Key', tm_order_col, value_col]].\
            set_index([sim_col, 'Group Key', tm_order_col])

        #obtain the list of indices
        rows = df_indexed.index.labels[0] #Students indices
        cols = df_indexed.index.labels[1] #Steps (Features) indices
        tms = df_indexed.index.labels[2] #tm_col values

        #obtain values for given indices of predictions and problem views
        vals = df_indexed.values.reshape(-1) #Answer Values

        #initialize the list to be inserted
        sparse_ftrs = []

        for t in np.unique(tms):
            #indices of the rows, cols, tms, vals list for the appropriate tm_order
            select_tm = (tms==t)
            sparse_ftrs.append(\
                coo_matrix((vals[select_tm], (rows[select_tm], cols[select_tm])),\
                shape=(m, n))
            )

        #add it to the object variable
        self.sparse_ftrs[ftr_name] = sparse_ftrs

    def assign_2D_sparse_ftrs_max(self, ftr_name, max_ftr_name):
        '''
        From self.sparse_ftrs[ftr_name], generate the same list but with the
        latest value based on tm_order (if at different tm_order a coordinate
        has multiple values, make it equal to the latest one at time t)
        Inputs:
            ftr_name: name of the 2D_sparse_ftrs in self.sparse_ftrs
            max_ftr_name: name of the new max_sparse_ftrs
        Outputs:
            self.sparse_ftrs[max_ftr_name]: this new sparse ftr will be appended
        '''
        #initialize a sparse empty running_X, get the ftrs to be maxed
        X = self.sparse_ftrs[ftr_name]
        running_X = coo_matrix(X[0].shape, dtype=np.int64).tocsc()
        X_max = []

        for t in range(len(X)):
            running_X[X[t].row, X[t].col] = X[t].data
            X_max.append(running_X.tocoo())

        self.sparse_ftrs[max_ftr_name] = X_max

    def load_2D_sparse_ftrs(self, ftr_name, filenames, foldername=None):
        '''
        if saved as pickle, load the 2D sparse ftr list (where list indices are
        time_order_col values)
        '''
        if isinstance(filenames, list):
            loaded_ftrs = []
            for name in filenames:
                loaded_ftrs.append(lu.load_pickle(name, foldername))

            loaded_ftrs = [val for sublist in loaded_ftrs for val in sublist]
            self.sparse_ftrs[ftr_name] = loaded_ftrs
        else:
            self.sparse_ftrs[ftr_name] = lu.load_pickle(filenames, foldername)

    def test_2D_sparse(self, Learner, tm_orders_totest, ftr_name, answers_name,
        fit_params, pred_params):
        '''
        Given a learner, test them iterativly on self.sparse_ftrs.
        Inputs:
            Learner: Class of the learner, expected methods of:
                fit(X=self.sparse_ftrs[ftr_name][t_cur])
                predict(X_answers=self.sprase_ftrs[answers_name][t_cur+1])
            tm_orders_totest: the number of tm_order's to iterate through
            ftr_name: name of the self.sparse_ftrs to use for fit
            answers_name: name of the self.sparse_ftrs that represents the
                answers to predict
            {fit_params}: additional parameters the learner may need for fit
            {pred_params}: additional parameters the learner may need for pred
        '''
        # import ipdb; ipdb.set_trace()
        #instantiate the learner
        learner = Learner()

        #lists to store results
        t_cur_results = []
        learner_name_results = []
        ftr_name_results = []
        answers_name_results = []
        tp_results = []
        tn_results = []
        fp_results = []
        fn_results = []
        acc_results = []

        for t_cur in range(tm_orders_totest):
            print('t_cur = {}'.format(t_cur))
            learner.fit(self.sparse_ftrs[ftr_name][t_cur], **fit_params)
            preds, actls = learner.pred(self.sparse_ftrs[answers_name][t_cur+1], **pred_params)

            #calculate TP, TN, FP, FN
            tp = np.sum(preds[actls==1] == actls[actls==1])
            tn = np.sum(preds[actls==0] == actls[actls==0])
            fp = np.sum(preds[actls==1] != actls[actls==1])
            fn = np.sum(preds[actls==0] != actls[actls==0])
            acc = (tp+tn)/(tp+tn+fp+fn)

            #insert results
            t_cur_results.append(t_cur)
            learner_name_results.append(str(learner))
            ftr_name_results.append(ftr_name)
            answers_name_results.append(answers_name)
            tp_results.append(tp)
            tn_results.append(tn)
            fp_results.append(fp)
            fn_results.append(fn)
            acc_results.append(acc)

        #store results in a dataframe
        results = pd.DataFrame({
            't_cur':t_cur_results,
            'Learner Name':learner_name_results,
            'ftr_name':ftr_name_results,
            'answers_name':answers_name_results,
            'TP':tp_results,
            'TN':tn_results,
            'FP':fp_results,
            'FN':fn_results,
            'Accuracy':acc_results
        })

        return results
