import numpy as np
import pandas as pd
import timeit
import time

from scipy.spatial import distance
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

import loading_util as lu

class Learner(object):
    def __init__(self, df=None):
        if df is None:
            df = pd.DataFrame()
        self.df = df

    def __str__(self):
        return 'Learner'

    def fit(self, X):
        print('using parent Learner fit method')

    def pred(self, X_answers):
        print('using parent Learner predict method')
        return np.zeros(0), np.zeros(0)

class NN_cos_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None, threshold=None):
        super().__init__(df)
        self.sim = sim
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'NN_cos_Learner|thresh={}'.format(self.threshold)

    def fit(self, X):
        '''
        Calculate the cosine similarity matrix
        Inputs:
            X: mxn coo_sparse matrix, m = each sample, n = # of features
        Ouputs:
            sim: mxm similarity dense matrix
        '''
        #what if an X of all 0's? it will default to 0 sim
        self.X = X
        self.sim = cosine_similarity(X, X)

    def pred(self, X_answers, threshold):
        '''
        using self.sim, predict on X_answers
        Inputs:
            X_answers: mxn coo_sparse matrix, m = each sample, n = # of features
                ***filled parts of the sparse matrix are what we are predicting
                for, labels are all binary 0/1 (False/True)
        Outputs:
            preds: ndarray of size filled_values in X_answers of the predictions
            actls: ndarray of size filled_values in X_answers of actual labels
        '''
        #assign threshold purely for labelling
        self.threshold = threshold

        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())

        #make weight_divisor so that it only includes weights that had a value
        weights_rows = self.X.row
        weights_cols = self.X.col
        weights_vals = np.ones(self.X.row.shape)
        weight_divisor_idx = coo_matrix((weights_vals, (weights_rows, weights_cols)),\
            shape=(pred_matrix.shape))
        weight_divisor = sparse.csc_matrix.dot(self.sim, weight_divisor_idx.tocsc())

        #divide by the appropriate sum of weights (only the ones used)
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        #store predictions for the appropriate X_answers indexes in a vector
        #and the actual values to return
        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class NN_cos_noselfsim_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None, threshold=None):
        super().__init__(df)
        self.sim = sim
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'NN_cos_noselfsim_Learner|thresh={}'.format(self.threshold)

    def fit(self, X):
        '''
        Calculate the cosine similarity matrix
        Inputs:
            X: mxn coo_sparse matrix, m = each sample, n = # of features
        Ouputs:
            sim: mxm similarity dense matrix
        '''
        #what if an X of all 0's? it will default to 0 sim
        self.X = X
        self.sim = cosine_similarity(X, X)

        #convert diagonal values to 0 to prevent the self student to contribute
        #to the prediction
        np.fill_diagonal(self.sim, 0)

    def pred(self, X_answers, threshold):
        '''
        using self.sim, predict on X_answers
        Inputs:
            X_answers: mxn coo_sparse matrix, m = each sample, n = # of features
                ***filled parts of the sparse matrix are what we are predicting
                for, labels are all binary 0/1 (False/True)
        Outputs:
            preds: ndarray of size filled_values in X_answers of the predictions
            actls: ndarray of size filled_values in X_answers of actual labels
        '''
        #assign threshold purely for labelling
        self.threshold = threshold

        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())

        #make weight_divisor so that it only includes weights that had a value
        weights_rows = self.X.row
        weights_cols = self.X.col
        weights_vals = np.ones(self.X.row.shape)
        weight_divisor_idx = coo_matrix((weights_vals, (weights_rows, weights_cols)),\
            shape=(pred_matrix.shape))
        weight_divisor = sparse.csc_matrix.dot(self.sim, weight_divisor_idx.tocsc())

        #divide by the appropriate sum of weights (only the ones used)
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        #store predictions for the appropriate X_answers indexes in a vector
        #and the actual values to return
        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class NN_cos_encode_wrong_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None, threshold=None, encode_wrong=None):
        super().__init__(df)
        self.sim = sim
        self.X = X
        self.threshold = threshold
        self.encode_wrong = encode_wrong

    def __str__(self):
        return 'NN_cos_encode_wrong_Learner|thresh={}|encode_wrong={}'.\
            format(self.threshold, self.encode_wrong)

    def fit(self, X, encode_wrong):
        '''
        Calculate the cosine similarity matrix
        Inputs:
            X: mxn coo_sparse matrix, m = each sample, n = # of features
        Ouputs:
            sim: mxm similarity dense matrix
        '''
        #purely for naming
        self.encode_wrong = encode_wrong

        #what if an X of all 0's? it will default to 0 sim
        self.X = X

        #change ecoded 0's to be encode_wrong
        data = self.X.data
        data[data==0] = encode_wrong
        self.X.data = data

        #change all labelled 0's to -1
        self.sim = cosine_similarity(X, X)

    def pred(self, X_answers, threshold):
        '''
        using self.sim, predict on X_answers
        Inputs:
            X_answers: mxn coo_sparse matrix, m = each sample, n = # of features
                ***filled parts of the sparse matrix are what we are predicting
                for, labels are all binary 0/1 (False/True)
        Outputs:
            preds: ndarray of size filled_values in X_answers of the predictions
            actls: ndarray of size filled_values in X_answers of actual labels
        '''
        #purely for labeling purposes
        self.threshold = threshold

        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())

        #make weight_divisor so that it only includes weights that had a value
        weights_rows = self.X.row
        weights_cols = self.X.col
        weights_vals = np.ones(self.X.row.shape)
        weight_divisor_idx = coo_matrix((weights_vals, (weights_rows, weights_cols)),\
            shape=(pred_matrix.shape))
        weight_divisor = sparse.csc_matrix.dot(self.sim, weight_divisor_idx.tocsc())

        #divide by the appropriate sum of weights (only the ones used)
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        #store predictions for the appropriate X_answers indexes in a vector
        #and the actual values to return
        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class NN_l1_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None, threshold=None):
        super().__init__(df)
        self.sim = sim
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'NN_cos_l1|thresh={}'.format(self.threshold)

    def fit(self, X):
        '''
        Calculate the cosine similarity matrix
        Inputs:
            X: mxn coo_sparse matrix, m = each sample, n = # of features
        Ouputs:
            sim: mxm similarity dense matrix
        '''
        # import ipdb; ipdb.set_trace()
        #what if an X of all 0's? it will default to 0 sim
        self.X = X

        #TODO How to handle missing values being treated as 0 in sparse?
        #determine the max and min of all possible X's to get range to subtract
        range_of_X = np.max(self.X.data) - np.min(self.X.data)

        #determine the size of each vector to normalize the distances
        norm = self.X.shape[1]

        #to get similarity, take range_of_X .- dist_matrix
        self.sim = range_of_X - (pairwise_distances(X.tocsr(), metric='l1', n_jobs=-1)/norm)

    def pred(self, X_answers, threshold):
        '''
        using self.sim, predict on X_answers
        Inputs:
            X_answers: mxn coo_sparse matrix, m = each sample, n = # of features
                ***filled parts of the sparse matrix are what we are predicting
                for, labels are all binary 0/1 (False/True)
        Outputs:
            preds: ndarray of size filled_values in X_answers of the predictions
            actls: ndarray of size filled_values in X_answers of actual labels
        '''
        #assign threshold purely for labelling
        self.threshold = threshold

        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())

        #make weight_divisor so that it only includes weights that had a value
        weights_rows = self.X.row
        weights_cols = self.X.col
        weights_vals = np.ones(self.X.row.shape)
        weight_divisor_idx = coo_matrix((weights_vals, (weights_rows, weights_cols)),\
            shape=(pred_matrix.shape))
        weight_divisor = sparse.csc_matrix.dot(self.sim, weight_divisor_idx.tocsc())

        #divide by the appropriate sum of weights (only the ones used)
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        #store predictions for the appropriate X_answers indexes in a vector
        #and the actual values to return
        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class NN_l1_encode_wrong_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None, threshold=None, encode_wrong=None):
        super().__init__(df)
        self.sim = sim
        self.X = X
        self.threshold = threshold
        self.encode_wrong = encode_wrong

    def __str__(self):
        return 'NN_l1_encode_wrong_Learner|thresh={}|encode_wrong={}'.\
            format(self.threshold, self.encode_wrong)

    def fit(self, X, encode_wrong):
        '''
        Calculate the cosine similarity matrix
        Inputs:
            X: mxn coo_sparse matrix, m = each sample, n = # of features
        Ouputs:
            sim: mxm similarity dense matrix
        '''
        #purely for naming
        self.encode_wrong = encode_wrong

        #what if an X of all 0's? it will default to 0 sim
        self.X = X

        #change ecoded 0's to be encode_wrong
        data = self.X.data
        data[data==0] = encode_wrong
        self.X.data = data

        #TODO How to handle missing values being treated as 0 in sparse?
        #determine the max and min of all possible X's to get range to subtract
        range_of_X = np.max(self.X.data) - np.min(self.X.data)

        #determine the size of each vector to normalize the distances
        norm = self.X.shape[1]

        #to get similarity, take range_of_X .- dist_matrix
        self.sim = range_of_X - (pairwise_distances(X.tocsr(), metric='l1', n_jobs=-1)/norm)

    def pred(self, X_answers, threshold):
        '''
        using self.sim, predict on X_answers
        Inputs:
            X_answers: mxn coo_sparse matrix, m = each sample, n = # of features
                ***filled parts of the sparse matrix are what we are predicting
                for, labels are all binary 0/1 (False/True)
        Outputs:
            preds: ndarray of size filled_values in X_answers of the predictions
            actls: ndarray of size filled_values in X_answers of actual labels
        '''
        #purely for labeling purposes
        self.threshold = threshold

        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())

        #make weight_divisor so that it only includes weights that had a value
        weights_rows = self.X.row
        weights_cols = self.X.col
        weights_vals = np.ones(self.X.row.shape)
        weight_divisor_idx = coo_matrix((weights_vals, (weights_rows, weights_cols)),\
            shape=(pred_matrix.shape))
        weight_divisor = sparse.csc_matrix.dot(self.sim, weight_divisor_idx.tocsc())

        #divide by the appropriate sum of weights (only the ones used)
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        #store predictions for the appropriate X_answers indexes in a vector
        #and the actual values to return
        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class NN_custom_withweights_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None, threshold=None):
        super().__init__(df)
        self.sim = sim
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'cust_dist|w_incor={}|w_cor={}|agg={}|thresh={}'\
            .format(self.w_incor, self.w_cor, self.agg, self.threshold)

    def create_sparse_stack(self, X_coo, agg, t_cur, save_name=None, load_name=None):
        '''
        Create the LHS and RHS of pairwise vector combinations between students,
        only capturing the upper triangular combaintions
        Inputs:
            X_coo: sparse 2D coo matrix where rows = stud, col = problem-step
            agg: string that determines how we aggregate the indep problem-step
                distances of correct and incorrect
            save_name: string of name when saving
        Outputs:
            lhs_stacked: LHS sparse 2D coo stack
            rhs_stacked: RHS sparse 2D coo stack
        '''
        #get number of students and problem steps for later
        m = X_coo.shape[0]
        n = X_coo.shape[1]

        if save_name is not None:
            saved_stacks = []

        #list for storing the upper triangular values created in each iter
        upper_tri_vals = []

        #convert X_coo to be csr for row slicing, counter for lhs
        X_csr = X_coo.tocsr()

        if load_name is not None:
            loaded_stacks = lu.load_pickle_workaround('{}_diff_tm{}'.format(load_name, t_cur), 'saved_diff_stacks')
            for diff in loaded_stacks:
                #agg based on string TODO VERIFY NO BUGS
                if agg == 'sum':
                    diff = diff.sum(axis=1)
                elif agg == 'avg':
                    diff = diff.mean(axis=1)
                elif agg == 'sqrt(sum_of_squares)':
                    diff.data = diff.data**2
                    diff = diff.sum(axis=1)
                    diff = diff**0.5
                else:
                    #default to sum
                    diff = diff.sum(axis=1)

                upper_tri_vals.append(diff)
        else:
            #loop [0,m-1) because don't need the final corner (since it is 0 in sim)
            for i in range(m-1):
                #get the appropriate slices
                lhs_slice = X_csr[i, :]
                lhs_data = np.tile(lhs_slice.data, (m-1)-i)
                lhs_cols = np.tile(lhs_slice.indices, (m-1)-i)
                lhs_rows = np.repeat(
                    np.arange(
                        start=0,
                        stop=(m-1)-i
                    ),
                    repeats=lhs_slice.data.shape[0]
                )
                lhs_slice = coo_matrix((lhs_data, (lhs_rows, lhs_cols)), shape=((m-1)-i, n))

                rhs_slice = X_csr[(i+1):, :]

                #take difference between slices, abs, then sum axis=1
                diff = lhs_slice.tocsr() - rhs_slice #csr is fastest from tests
                diff.data = np.absolute(diff.data)

                if save_name is not None: #save before the aggregation
                    saved_stacks.append(diff)

                #agg based on string
                if agg == 'sum':
                    diff = diff.sum(axis=1)
                elif agg == 'avg':
                    diff = diff.mean(axis=1)
                elif agg == 'sqrt(sum_of_squares)':
                    diff.data = diff.data**2
                    diff = diff.sum(axis=1)
                    diff = np.asarray(diff)**0.5
                else:
                    #default to sum
                    diff = diff.sum(axis=1)

                upper_tri_vals.append(diff)

        #convert to a long 1D numpy array
        upper_tri_vals = np.concatenate(upper_tri_vals)

        if save_name is not None:
            lu.save_pickle_workaround('{}_diff_tm{}'.format(save_name, t_cur), \
                saved_stacks, 'saved_diff_stacks')

        return upper_tri_vals

    def fit(self, t_cur, X_correct_latest, X_correct_cnt_latest, X_incorrect_cnt_latest, w_correct, w_incorrect, agg, load, save):
        '''
        Calculate similarity matrix based on the correct_cnt_latest and
        incorrect_cnt_latest
        Inputs:
            t_cur: current tm
            X_correct_latest: 2D sparse coo holding the current 0/1 correct/not
                at t_cur, here it is set to self.X purely for pred() to calc
                predictions given the similarity that will be generated here
            X_correct_cnt_latest: 2D sparse coo holding the total corrects for a
                student currently
            X_incorrect_cnt_latest: same as above but for incorrects
            w_correct:
            w_incorrect:
        Ouputs:
            self.sim: mxm similarity dense matrix using custom similarity
        '''
        '''UNCOMMENT ENCLOSED FOR TESTING'''
        # import ipdb; ipdb.set_trace()
        # X_correct_latest = np.zeros(1)
        #
        # data_cor = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1])
        # rows_cor = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
        # cols_cor = np.array([0, 3, 1, 2, 0, 1, 0, 1, 2, 1, 2, 3])
        # X_correct_cnt_latest = coo_matrix(
        #     (data_cor, (rows_cor, cols_cor)), shape=(5,4)
        # )
        #
        # data_incor = np.array([5, 1, 2, 2, 3, 1, 4, 5, 1, 1, 1, 1, 1, 1, 2])
        # rows_incor = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4])
        # cols_incor = np.array([0, 2, 3, 0, 1, 2, 3, 0, 2, 3, 0, 3, 1, 2, 3])
        # X_incorrect_cnt_latest = coo_matrix(
        #     (data_incor, (rows_incor, cols_incor)), shape=(5,4)
        # )
        '''END TESTING'''
        #for labelling
        self.w_cor = w_correct
        self.w_incor = w_incorrect
        self.agg = agg

        #for pred
        self.X = X_correct_latest

        #get size of student x problem-step sparse matrix
        m = X_correct_cnt_latest.shape[0]
        n = X_correct_cnt_latest.shape[1]

        #initialize the dist matrix to be dense zeros size stud x stud
        dist = np.zeros((m, m))

        #if any student has gotten a problem-step correct, it is 1, else 0
        X_correct_cnt_latest.data = X_correct_cnt_latest.data > 0

        #TODO put in a method
        if load:
            save_name_cor = None
            save_name_incor = None
            load_name_cor = 'cor'
            load_name_incor = 'incor'
        else:
            save_name_cor = 'cor'
            save_name_incor = 'incor'
            load_name_cor = None
            load_name_incor = None
        if not(save):
            save_name_cor = None
            save_name_incor = None
            load_name_cor = None
            load_name_incor = None

        #create the LHS and RHS stacks for correct and incorrects as coo sparses TODO change load_name here
        cor_dist = self.create_sparse_stack(X_correct_cnt_latest, agg, t_cur, save_name_cor, load_name_cor)
        incor_dist = self.create_sparse_stack(X_incorrect_cnt_latest, agg, t_cur, save_name_incor, load_name_incor)

        #multiply by weights, add them together for distance, turn into upper
        #triangular starting from the first spot NOT in diagonal
        dist_vals = np.asarray((w_correct*cor_dist) + (w_incorrect*incor_dist)).reshape(-1)
        dist[np.triu_indices(n=m, k=1)] = dist_vals
        dist[np.tril_indices(n=m, k=-1)] = dist_vals

        #create a simlarity matrix based on the dist matrix bounded to [0,1]
        #max(dist) is the same as range(dist) in this case as all dist values>0
        self.sim = 1.0 - (dist/np.max(dist))
        np.fill_diagonal(self.sim, 0)

    def pred(self, X_answers, threshold):
        '''
        using self.sim, predict on X_answers
        Inputs:
            X_answers: mxn coo_sparse matrix, m = each sample, n = # of features
                ***filled parts of the sparse matrix are what we are predicting
                for, labels are all binary 0/1 (False/True)
        Outputs:
            preds: ndarray of size filled_values in X_answers of the predictions
            actls: ndarray of size filled_values in X_answers of actual labels
        '''
        #assign threshold purely for labelling
        self.threshold = threshold

        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())

        #make weight_divisor so that it only includes weights that had a value
        weights_rows = self.X.row
        weights_cols = self.X.col
        weights_vals = np.ones(self.X.row.shape)
        weight_divisor_idx = coo_matrix((weights_vals, (weights_rows, weights_cols)),\
            shape=(pred_matrix.shape))
        weight_divisor = sparse.csc_matrix.dot(self.sim, weight_divisor_idx.tocsc())

        #divide by the appropriate sum of weights (only the ones used)
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        #store predictions for the appropriate X_answers indexes in a vector
        #and the actual values to return
        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class Uniform_Random_Learner(Learner):
    def __init__(self, df=None, X=None, threshold=None):
        super().__init__(df)
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'Uniform_Random_Learner|thresh={}'.format(self.threshold)

    def pred(self, X_answers, threshold):
        #purely for naming purposes
        self.threshold = threshold

        preds = np.random.uniform(0, 1, X_answers.data.shape) >= threshold
        actls = X_answers.data

        return preds, actls

class Gloabl_Avg_Learner(Learner):
    def __init__(self, df=None, X=None, threshold=None):
        super().__init__(df)
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'Gloabl_Avg_Learner|thresh={}'.format(self.threshold)

    def fit(self, X):
        self.X = X

    def pred(self, X_answers, threshold):
        #purely for naming purposes
        self.threshold = threshold

        preds = np.ones(X_answers.data.shape)*np.mean(self.X.data) >= threshold
        actls = X_answers.data

        return preds, actls

class Within_XCol_Avg_Learner(Learner):
    def __init__(self, df=None, X=None, threshold=None):
        super().__init__(df)
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'Within_XCol_Avg_Learner|thresh={}'.format(self.threshold)

    def fit(self, X):
        self.X = X

    def pred(self, X_answers, threshold):
        #purely for naming purposes
        self.threshold = threshold

        '''predictions for X_answers are the average of the columns in self.X'''
        #first find matrix size X.shape that holds 1 for where there is a value
        #in the sparse self.X
        X_val_idx = coo_matrix((np.ones(self.X.data.shape), (self.X.row, self.X.col)),\
            shape=(self.X.shape))

        #sum across columns (by collapsing rows) to see how many elements are
        #truly in each column of the sparse self.X
        X_col_counts = X_val_idx.sum(axis=0)

        #now get the mean for each col
        X_col_means = np.asarray(self.X.sum(axis=0)/X_col_counts).reshape(-1)

        #assign nan values (cols that didn't have any vals in self.X) to glb_avg
        X_col_means[np.isnan(X_col_means)] = np.mean(self.X.data)

        #get a predictions matrix by extending X_col_means by the number of rows
        #in self.X
        pred_matrix = np.tile(X_col_means, self.X.shape[0]).reshape(self.X.shape)

        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class Within_XRow_Avg_Learner(Learner):
    def __init__(self, df=None, X=None, threshold=None):
        super().__init__(df)
        self.X = X
        self.threshold = threshold

    def __str__(self):
        return 'Within_XRow_Avg_Learner|thresh={}'.format(self.threshold)

    def fit(self, X):
        self.X = X

    def pred(self, X_answers, threshold):
        '''predictions for X_answers are the average of the columns in self.X'''
        #purely for naming purposes
        self.threshold = threshold

        #first find matrix size X.shape that holds 1 for where there is a value
        #in the sparse self.X
        X_val_idx = coo_matrix((np.ones(self.X.data.shape), (self.X.row, self.X.col)),\
            shape=(self.X.shape))

        #sum across rows (by collapsing cols) to see how many elements are truly
        #in each row of the sparse self.X
        X_row_counts = X_val_idx.sum(axis=1)

        #now get the mean for each row
        X_row_means = np.asarray(self.X.sum(axis=1)/X_row_counts).reshape(-1)

        #assign nan values (rows that didn't have any vals in self.X) to glb_avg
        X_row_means[np.isnan(X_row_means)] = np.mean(self.X.data)

        #get a prediction matrix by extending X_row_means by the number of cols
        #in self.X
        pred_matrix = np.repeat(X_row_means, self.X.shape[1]).reshape(self.X.shape)

        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls
