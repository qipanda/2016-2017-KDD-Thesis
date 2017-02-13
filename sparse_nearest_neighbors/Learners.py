import numpy as np
import pandas as pd

from scipy.spatial import distance
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

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

    def predict(self, X_answers):
        print('using parent Learner predict method')
        return np.zeros(0), np.zeros(0)

class NN_cos_Learner(Learner):
    def __init__(self, df=None, X=None, sim=None):
        super().__init__(df)
        self.sim = sim
        self.X = X

    def __str__(self):
        return 'NN_Learner'

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
        #assign diagonals to 1 to avoid dividing by 0
        # np.fill_diagonal(self.sim, 1)

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
        #get the prediction matrix that will be (mxm)dot(mxn) giving (mxn)
        pred_matrix = sparse.csc_matrix.dot(self.sim, self.X.tocsc())
        weight_divisor = np.repeat(np.sum(self.sim, axis=1), pred_matrix.shape[1]).reshape(-1, pred_matrix.shape[1])
        pred_matrix = pred_matrix/weight_divisor

        #if any nan's from 0 division fill with the average
        pred_matrix[np.isnan(pred_matrix)] = np.mean(self.X.data)

        preds = pred_matrix[X_answers.row, X_answers.col] >= threshold
        actls = X_answers.data

        return preds, actls

class Uniform_Random_Learner(Learner):
    def __str__(self):
        return 'Uniform_Random_Learner'

    def predict(self, pred_df, pred_col):
        '''
        this subclass simply takes in a dataframe, and predicts randomly [0,1] and
        calls the new column pred_col + '_pred'. No need to fit anything here
        '''
        pred_df[pred_col + '_pred'] = np.random.uniform(0, 1, pred_df.shape[0])
        self.predictions = pred_df

        return self.predictions.copy(), pred_col, pred_col+'_pred'

class Gloabl_Avg_Learner(Learner):
    def __str__(self):
        return 'Gloabl_Avg_Learner'

    def predict(self, pred_df, pred_col):
        '''
        this subclass simply takes the global average of the pred_col and uses that
        as all predictions from the train_df
        '''
        pred_df[pred_col + '_pred'] = self.train_df[pred_col].mean()
        self.predictions = pred_df

        return self.predictions.copy(), pred_col, pred_col+'_pred'

class Within_Col_Avg_Learner(Learner):
    def __init__(self, df=None, within_col=None):
        super().__init__(df)
        if within_col is None:
            within_col = 'NONE ASSIGNED'
        self.within_col = within_col

    def __str__(self):
        if isinstance(self.within_col, str):
            return 'Within_Col_Avg_Learner | within_col: ' + self.within_col
        elif isinstance(self.within_col, list):
            return 'Within_Col_Avg_Learner | within_col: ' + str(self.within_col)

    def predict(self, pred_df, pred_col):
        '''
        this subclass predicts based on the average of all values in each within_col
        Can handle within_col's that are singular (just string) or multiple columns
        (inputted as a list of strings)
        '''
        within_means = self.train_df.groupby(self.within_col).agg({pred_col:pd.Series.mean})

        within_means = within_means.reset_index()
        if isinstance(self.within_col, str):
            within_means.columns = [self.within_col, pred_col + '_pred']
        elif isinstance(self.within_col, list):
            within_means.columns = [*self.within_col, pred_col + '_pred']

        pred_df = pd.merge(pred_df, within_means, on=self.within_col, how='left')
        self.predictions = pred_df

        return self.predictions.copy(), pred_col, pred_col+'_pred'
