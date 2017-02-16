import numpy as np
import pandas as pd

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
        return 'Within_XRow_Avg_Learner'

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
