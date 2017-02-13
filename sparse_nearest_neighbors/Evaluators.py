import pandas as pd
import numpy as np

class BinClass_Evaluator(object):
    '''
    This is meant to hold two numpy arrays and compute various accuracy measures
    between them for binary classification task

    *Excludes rows that have NULL/NaN/etc. in actl, counts rows that have
    NULL/NaN/etc. in pred as INCORRECT
    '''

    def __init__(self, prediction_df, actl_col, pred_col, macro_pred_group=None):
        self.prediction_df = prediction_df
        self.actl_col = actl_col
        self.pred_col = pred_col
        self.macro_pred_group = macro_pred_group

    def __str__(self):
        if self.macro_pred_group is None:
            return 'BinClass_Evaluator'
        else:
            return 'BinClass_Evaluator_MACRO: ' + self.macro_pred_group

    def filter(self):
        '''
        Excludes rows that have NULL/NaN/etc. in actl, counts rows that have
        NULL/NaN/etc. in pred as INCORRECT
        '''
        if self.macro_pred_group is None:
            #assign the vectors of micro predictions and actual values
            self.actl = self.prediction_df[self.actl_col].values
            self.pred = self.prediction_df[self.pred_col].values

            #exclude rows where no actual prediction
            self.pred = self.pred[~np.isnan(self.actl)]
            self.actl = self.actl[~np.isnan(self.actl)]

            #rows where prediction is nan, count as incorrect by assigning opposite value
            self.pred[np.isnan(self.pred)] = ~(self.actl[np.isnan(self.pred)]==1)

        else:
            #exclude rows with no actual pred_col value
            self.prediction_df = self.prediction_df.loc[~self.prediction_df[self.actl_col].isnull()]
            #count rows with nan predictions as incorrect by assigning the opposite
            self.prediction_df.loc[self.prediction_df[self.pred_col].isnull(), self.pred_col] =\
                ((self.prediction_df[self.actl_col] != 1)*1)


    def evaluate(self):
        # print('using parent BinClass_Evaluator evaluate method')
        return self.pred == self.actl

class RMSE_BinClass_Evaluator(BinClass_Evaluator):
    def __str__(self):
        if self.macro_pred_group is None:
            return 'RMSE_BinClass_Evaluator'
        else:
            return 'RMSE_BinClass_Evaluator_MACRO: ' + self.macro_pred_group

    def evaluate(self):
        '''
        Calculates the overall RMSE between pred and actl numpy arrays.

        Calculation:
            SQRT(SUM_i((y_pred_i - y_actl_i)^2)/samples)
        '''
        self.filter()
        if self.macro_pred_group is None:
            return np.sqrt(np.sum((self.pred - self.actl)**2)/self.pred.shape[0])

        else: #for macro average, calculate RMSE for each self.macro_pred_group, then average those
            self.prediction_df['micro_evals'] = \
                (self.prediction_df[self.pred_col] - self.prediction_df[self.actl_col])**2
            return np.mean(np.sqrt(self.prediction_df.groupby(self.macro_pred_group)['micro_evals'].mean().values))


class Acc_BinClass_Evaluator(BinClass_Evaluator):
    def __str__(self):
        if self.macro_pred_group is None:
            return 'Acc_BinClass_Evaluator'
        else:
            return 'Acc_BinClass_Evaluator_MACRO: ' + self.macro_pred_group

    def evaluate(self, thresh):
        '''
        Calculates the overall accuracy of a binary classification task given
        a threshold (predict 1 if >= threshold, 0 otherwise)
        '''
        self.filter()
        if self.macro_pred_group is None:
            return np.sum((self.pred >= thresh) == self.actl)/self.pred.shape[0]
        else:
            self.prediction_df['micro_evals'] = \
                (self.prediction_df[self.pred_col] >= thresh) == self.prediction_df[self.actl_col]
            return np.mean(self.prediction_df.groupby(self.macro_pred_group)['micro_evals'].mean().values)
