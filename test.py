import pdb
import numpy as np
import pandas as pd
import random

def star(a, b, c, d):
    return {'lol':a+b, 'haha':c+d}

def main():
    # pdb.set_trace()
    # a=1
    # test = star(1,2,3,4)
    df = pd.DataFrame([1,2,3])
    pdb.set_trace()
    for row in df.itertuples():
        print(row)

if __name__ == "__main__":
    main()








def createValidationSet(dataset, max_iter, prct_of_data, problem_thresh_prct, problem_thresh):
    #problem_thresh means the % of data(row count) we want in the val_set relative to total data returned
    #initialize valset and trainset, from testing problem_thresh_prct should be <= 0.5 for prct_of_data = 0.2
    val_set = pd.DataFrame(columns=dataset.columns)
    train_set = dataset

    #create a validation set as a % of whole test set, first trim columns to what is needed
    #dataset with only >= problem_thresh
    trimmed_set = removeLowerThanProblemThresh(['Anon Student Id', 'Unit'], 'Problem Name', \
    'Unique Problems', train_set, problem_thresh)

    #get each Student-Unit_problem by their min(), list is sorted within each Student-Unit by timestamp
    #First Transaction Time timestamp in a problem
    min_tm_set = studUnitProbTimewise(['Anon Student Id', 'Unit', 'Problem Name'], 'First Transaction Time', \
    'MIN_Trans_TM', trimmed_set)

    iters = 0
    #While the val_set has not not exceeded or equaled threshold of total data retained
    while ((val_set.shape[0]/(val_set.shape[0]+train_set.shape[0])) < prct_of_data) and (iters<max_iter):
        for stud_unit_comb in (trimmed_set[['Anon Student Id', 'Unit']].drop_duplicates().values.tolist()): #THIS LOOP IS INEFFICIENT AFTER 1ST ITER!!!
            #temp_set are all steps/probs from a given stud_unit_comb that meets problem_thresh
            #rand_question is a random problem from this stud_unit_comb
            temp_set = min_tm_set[\
            (min_tm_set['Anon Student Id']==stud_unit_comb[0])&(min_tm_set['Unit']==stud_unit_comb[1])]

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

            #remove all steps in this stud_unit_comb that relate to rand_questions
            #AND happened AFTER rand_question, first create a dataframe with these rows to remove
            #can use index[0] because we reset it above
            remove_set = min_tm_set[\
            (min_tm_set['Anon Student Id']==rand_question['Anon Student Id'].ix[0])\
            &(min_tm_set['Unit']==rand_question['Unit'].ix[0])\
            &(min_tm_set['MIN_Trans_TM']>=rand_question['MIN_Trans_TM'].ix[0])]

            #remove step from train_set (leave only whats NOT indexed in remove_set
            train_set = pd.merge(train_set, remove_set, how='left', \
            on = ['Anon Student Id', 'Unit', 'Problem Name'])
            #all nulls are values that are NOT in remove set, so keep them, drop TM col after use
            train_set = train_set[train_set['MIN_Trans_TM'].isnull()]
            train_set = train_set.drop('MIN_Trans_TM', axis=1)

            #add rand_question rows to val set
            val_set = val_set.append(\
            pd.merge(trimmed_set, rand_question, how='inner',\
            on=['Anon Student Id', 'Unit', 'Problem Name'])[trimmed_set.columns]\
            )

        #create a validation set as a % of whole test set, first trim columns to what is needed
        #dataset with only >= problem_thresh
        trimmed_set = removeLowerThanProblemThresh(['Anon Student Id', 'Unit'], 'Problem Name', \
        'Unique Problems', train_set, problem_thresh)

        #get each Student-Unit_problem by their min(), list is sorted within each Student-Unit by timestamp
        #First Transaction Time timestamp in a problem
        min_tm_set = studUnitProbTimewise(['Anon Student Id', 'Unit', 'Problem Name'], 'First Transaction Time', \
        'MIN_Trans_TM', trimmed_set)

        print('Finished Validation iter ', iters)
        iters += 1

    #reset val_set index
    val_set = val_set.reset_index(drop=True)

    #calculate loss of data
    rows_retained = train_set.shape[0] + val_set.shape[0]
    rows_lost = dataset.shape[0] - train_set.shape[0] - val_set.shape[0]
    prct_lost = rows_lost/dataset.shape[0]
    prct_ratio_actl = (val_set.shape[0]/(val_set.shape[0]+train_set.shape[0]))

    return val_set, train_set, rows_retained, rows_lost, prct_lost, prct_ratio_actl
