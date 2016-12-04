import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb #debugger
import time

def importText(fileLocation, seperator, allrows, numrows=1):
    if allrows:
        return pd.read_csv(fileLocation, sep=seperator)
    else:
        return pd.read_csv(fileLocation, sep=seperator, nrows=numrows)

def splitCol(dataset, col_name, delim, new_col_names):
    #splits a column in a pandas dataframe and returns it with new names
    #.str.split creates vales into lists based on split, apply Series to turn lists into horiz.
    #vectors for each row
    t0 = time.time()
    appendset = dataset[col_name].str.split(delim, expand=True) #expand=True makes it a dataframe;
    print("inside splitCol, split at: ", time.time()-t0)
    appendset.columns = new_col_names
    print("inside splitCol, new col names at: ", time.time()-t0)
    splitset = dataset.drop(col_name, 1)
    print("inside splitCol, drop col at: ", time.time()-t0)
    splitset = splitset.join(appendset)
    print("inside splitCol, joined at: ", time.time()-t0)

    return splitset

def mergeCols(dataset, cols_to_merge, delim, new_col_name, drop_old):
    dataset[new_col_name] = dataset[cols_to_merge].astype(str).apply(lambda x: delim.join(x), axis=1)

    if drop_old:
        dataset = dataset.drop(cols_to_merge, axis=1)

    return dataset

def changeColsToDtime(dataset, col_names):
    dataset[col_names] = dataset[col_names].apply(pd.to_datetime) #COMPUTATIONLY SLOW HERE
    return dataset

def removeLowerThanProblemThresh(groupby_key, count_col, count_name, dataset, problem_thresh):
    #clean data to only have rows with student-unit's that have >= problem thresh problems
    #Count unique Problem's for Stud-Unit pairs
    filterset = dataset.groupby(groupby_key).\
    apply(lambda x: x[count_col].nunique())\
    .reset_index().rename(columns={0:count_name}) #makes index combinations again instead of multiindex hierarchical

    #filter down to only student-unit pairs with >= problem_thresh Count
    filterset = filterset[filterset[count_name] >= problem_thresh]

    #workset inner join filterset on student-unit
    dataset = pd.merge(dataset, filterset, how='inner', on=groupby_key)[dataset.columns]

    return dataset #this is the original data but with Problem Hierarchy split and filtered for threshold of problems

def studUnitProbTimewise(groupby_key, tm_col, min_tm_col, trimmed_set):
    #rank every problem within student-unit by their first trans time
    min_tm_set = trimmed_set.groupby(groupby_key).\
    apply(lambda x: x[tm_col].min())\
    .reset_index().rename(columns={0:min_tm_col})

    #sort in ascending timestamp within the groupby
    min_tm_set = min_tm_set.sort_values(by=min_tm_col, ascending=True)

    return min_tm_set

def createValidationSet(dataset, max_iter, prct_of_data, problem_thresh_prct, problem_thresh):
    #problem_thresh means the % of data(row count) we want in the val_set relative to total data returned
    #initialize valset and trainset, from testing problem_thresh_prct should be <= 0.5 for prct_of_data = 0.2

    val_set = pd.DataFrame(columns=dataset.columns)
    test_set = pd.DataFrame(columns=dataset.columns)
    train_set = dataset
    iters = 0

    t0 = time.time()
    while True:
        #create a validation set as a % of whole test set, first trim columns to what is needed
        #dataset with only >= problem_thresh
        trimmed_set = removeLowerThanProblemThresh(['Anon Student Id', 'Unit', 'Section'], 'Problem Name', \
        'Unique Problems', train_set, problem_thresh)

        #get each Student-Unit-Section_problem by their min(), list is sorted within each Student-Unit by timestamp
        #First Transaction Time timestamp in a problem
        min_tm_set = studUnitProbTimewise(['Anon Student Id', 'Unit', 'Section', 'Problem Name'], 'First Transaction Time', \
        'MIN_Trans_TM', trimmed_set)

        for stud_unit_section in (trimmed_set[['Anon Student Id', 'Unit', 'Section']].drop_duplicates().values.tolist()):
            #temp_set are all steps/probs from a given stud_unit_section that meets problem_thresh
            #rand_question is a random problem from this stud_unit_section
            temp_set = min_tm_set[\
            (min_tm_set['Anon Student Id']==stud_unit_section[0])&(min_tm_set['Unit']==stud_unit_section[1])\
            &(min_tm_set['Section']==stud_unit_section[2])]

            #how many rows to exclude
            rows_buffered = math.ceil(problem_thresh_prct*temp_set.shape[0])
            rows_considered = temp_set.shape[0] - rows_buffered

            #check if no rows to be considered
            if rows_considered == 0:
                break

            #only choose from back 100-problem_thresh_prct of rows with equal chance using weights
            rand_question = temp_set.sample(\
            weights=np.append(np.zeros(rows_buffered), np.ones(rows_considered)*(1./rows_considered))\
            ).reset_index(drop=True)

            #remove all steps in this stud_unit_section that relate to rand_questions
            #AND happened AFTER rand_question, first create a dataframe with these rows to remove
            #can use index[0] because we reset it above
            remove_set = min_tm_set[\
            (min_tm_set['Anon Student Id']==rand_question['Anon Student Id'].ix[0])\
            &(min_tm_set['Unit']==rand_question['Unit'].ix[0])\
            &(min_tm_set['Section']==rand_question['Section'].ix[0])\
            &(min_tm_set['MIN_Trans_TM']>=rand_question['MIN_Trans_TM'].ix[0])]

            #remove step from train_set (leave only whats NOT indexed in remove_set
            train_set = pd.merge(train_set, remove_set, how='left', \
            on = ['Anon Student Id', 'Unit', 'Section', 'Problem Name'])
            #all nulls are values that are NOT in remove set, so keep them, drop TM col after use
            train_set = train_set[train_set['MIN_Trans_TM'].isnull()]
            train_set = train_set.drop('MIN_Trans_TM', axis=1)

            #add rand_question rows to val set
            val_set = val_set.append(\
            pd.merge(trimmed_set, rand_question, how='inner',\
            on=['Anon Student Id', 'Unit', 'Section', 'Problem Name'])[trimmed_set.columns]\
            )

            #add a random problem (if any) after random one to the test set
            if remove_set.shape[0] > 1:
                try:
                    test_question = min_tm_set[\
                    (min_tm_set['Anon Student Id']==rand_question['Anon Student Id'].ix[0])\
                    &(min_tm_set['Unit']==rand_question['Unit'].ix[0])\
                    &(min_tm_set['Section']==rand_question['Section'].ix[0])\
                    &(min_tm_set['MIN_Trans_TM']>rand_question['MIN_Trans_TM'].ix[0])].sample()
                except ValueError:
                    #skip if none!
                    print('in data_manip, none found > tm than random_question!')

                test_set = test_set.append(\
                pd.merge(trimmed_set, test_question, how='inner',\
                on=['Anon Student Id', 'Unit', 'Section', 'Problem Name'])[trimmed_set.columns]\
                )

        print('IN DATA_MANIP, finished validation iter ', iters, ' t=', time.time()-t0)
        iters += 1

        #While the val_set has not not exceeded or equaled threshold of total data retained
        if ((val_set.shape[0]/(val_set.shape[0]+train_set.shape[0])) >= prct_of_data) or (iters >= max_iter):
            break

    #reset val_set index
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    #calculate loss of data
    rows_retained = train_set.shape[0] + val_set.shape[0] + test_set.shape[0]
    rows_lost = dataset.shape[0] - train_set.shape[0] - val_set.shape[0] - test_set.shape[0]
    prct_lost = rows_lost/dataset.shape[0]
    prct_ratio_val_to_train = (val_set.shape[0]/(val_set.shape[0]+train_set.shape[0]))

    return train_set, val_set, test_set, rows_retained, rows_lost, prct_lost, prct_ratio_val_to_train

def SVDNullReplace(row, default_col, cond_col, default_C):
    if(np.isnan(row[cond_col])):
        #if not in trainset, it is unknown for the purpose of training
        return default_C
    else:
        return row[default_col]

def sigmoid(z):
    try:
        ans = 1/(1+math.exp(-z))
    except OverflowError:
        ans = float('inf')

    return ans
