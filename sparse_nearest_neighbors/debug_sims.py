import numpy as np
import matplotlib.pyplot as plt
import loading_util as lu

'''
This script is meant to compare the {-1,0,+1} method with the custom similarity
method by looking at examples of the worst, average, and best matched students
'''

def hist_sim(sim, bins_hist, range_hist, xlabel, xtick, ylabel, title):
    #create new figure
    plt.figure()

    #select the upper triangular part of the sim matrix ecluding the diag (k=1)
    flattend_sim = sim[np.triu_indices(n=sim.shape[0], k=1)]

    #find the 5th, 50th, and 95th percentile values
    low = np.percentile(flattend_sim, 1, interpolation='nearest')
    mid = np.percentile(flattend_sim, 50, interpolation='nearest')
    high = np.percentile(flattend_sim, 99, interpolation='nearest')

    #plot histogram
    plt.hist(x=flattend_sim, bins=bins_hist, range=range_hist, label='normal_hist')
    plt.hist(x=flattend_sim, bins=bins_hist, range=range_hist, \
        cumulative=True, histtype='step', label='cumulative')
    plt.axvline(x=low, label='1st percentile = {}'.format(round(low, 2)), linewidth=0.5, linestyle='--', color='r')
    plt.axvline(x=mid, label='50th percentile = {}'.format(round(mid, 2)), linewidth=0.5, linestyle='--', color='y')
    plt.axvline(x=high, label='99th percentile = {}'.format(round(high, 2)), linewidth=0.5, linestyle='--', color='g')

    plt.xlabel(xlabel)
    plt.xticks(xtick)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

    return low, mid, high

if __name__ == '__main__':
    '''1.)load the sims and X's'''
    sim_negencode = lu.load_pickle('encode_neg1_zero_pos1_sim_tm18', 'debugging_sims')
    sim_cust_sum = lu.load_pickle('encode_cust_sum_tm18', 'debugging_sims')
    sim_cust_avg = lu.load_pickle('encode_cust_avg_tm18', 'debugging_sims')
    sim_cust_sqrtsum = lu.load_pickle('encode_cust_sqrtsum_tm18', 'debugging_sims')

    X_correct_latest = lu.load_pickle('X_correct_latest_tm18', 'debugging_sims')
    data = X_correct_latest.data
    data[data==0] = -1
    X_correct_latest.data = data #encode the -1

    X_correct_cnt_latest = lu.load_pickle('X_correct_cnt_latest_tm18', 'debugging_sims')
    X_correct_cnt_latest.data = X_correct_cnt_latest.data > 0 #encode T/F
    X_incorrect_cnt_latest = lu.load_pickle('X_incorrect_cnt_latest_tm18', 'debugging_sims')

    '''2.)Create a histogram of the similarity scores for each method'''
    sims = [sim_negencode, sim_cust_sum, sim_cust_avg, sim_cust_sqrtsum]
    bins_hists = [100]*4
    range_hists = [(-0.2, 0.2)] + [(0.0, 1.0)]*3
    xlabels = ['sim_score']*4
    x_ticks = [(-2 + np.arange(5))/10] + [np.arange(11)/10]*3
    ylabels = ['freq']*4
    titles = ['{-1, 0, +1} encoding', 'cust encoding: sum', \
        'cust encoding: avg', 'cust encoding: sqrtsum']
    percentile_vals = []

    gen = zip(sims, bins_hists, range_hists, xlabels, x_ticks, ylabels, titles)
    for sim, bins_hist, range_hist, xlabel, xtick, ylabel, title in gen:
        low, mid, high = hist_sim(sim, bins_hist, range_hist, xlabel, xtick, ylabel, title)
        percentile_vals.append([low, mid, high])

    '''3.)Analyze pairs of vectors that are 1st, 50th, 99th percentile'''
    #for {-1, 0, +1} first, show each vector and then their difference
    import ipdb; ipdb.set_trace()

    fig, axarr = plt.subplots(nrows=3, ncols=3)
    for j, prct in enumerate(percentile_vals[0]):
        #find the indices of this prct
        sim = sims[0]
        idx = np.argwhere(sim==prct)[0, :]

        #take the two slices our from the X feature matrix with the coressponding prct.
        stud_A = X_correct_latest.tocsr()[idx[0], :].toarray().reshape(-1)
        stud_B = X_correct_latest.tocsr()[idx[1], :].toarray().reshape(-1)
        x_range = np.arange(stud_A.shape[0])

        axarr[0, j].plot(x_range, stud_A, label='stud_A')
        axarr[1, j].plot(x_range, stud_B, label='stud_B')
        axarr[2, j].plot(x_range, (stud_A - stud_B), label='stud_A - stud_B')

    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.title('Student Vector Pair Samples for {-1, 0, +1} encoding')
    plt.xlabel('{1st || 50th || 99th} percentile similarity student pairs')
    plt.ylabel('{Stud_A - Stud_B || Stud_B || Stud_A}')
    plt.show()

    #for cust sims corrects
    titles = ['cust encoding: sum', 'cust encoding: avg', 'cust encoding: euclidean']

    gen = zip(titles, sims[1:], percentile_vals[1:])
    for title, sim, prcts in gen:
        fig_cor, axarr_cor = plt.subplots(nrows=3, ncols=3)
        for j, prct in enumerate(prcts):
            #find the indices of this prct
            idx = np.argwhere(sim==prct)[0, :]

            #for corrects first
            stud_A_cor = X_correct_cnt_latest.tocsr()[idx[0], :].toarray().reshape(-1)
            stud_B_cor = X_correct_cnt_latest.tocsr()[idx[1], :].toarray().reshape(-1)

            #in general the x_range is number of problem-steps
            x_range = np.arange(stud_A_cor.shape[0])

            #create plots of vector pairs for cor and incor
            axarr_cor[0, j].plot(x_range, stud_A_cor, label='stud_A_cor')
            axarr_cor[1, j].plot(x_range, stud_B_cor, label='stud_B_cor')
            axarr_cor[2, j].plot(x_range, (stud_A_cor - stud_B_cor), label='stud_A_cor - stud_B_cor')

        # add a big axes, hide frame
        fig_cor.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.title('Student Vector Pair Samples for {}: Correct T/F'.format(title))
        plt.xlabel('{1st || 50th || 99th} percentile similarity student pairs')
        plt.ylabel('{Stud_A - Stud_B || Stud_B || Stud_A}')
        plt.show()

    #for cust sims incorrects
    gen = zip(titles, sims[1:], percentile_vals[1:])
    for title, sim, prcts in gen:
        fig_incor, axarr_incor = plt.subplots(nrows=3, ncols=3)
        for j, prct in enumerate(prcts):
            #find the indices of this prct
            idx = np.argwhere(sim==prct)[0, :]

            #for corrects first
            stud_A_incor = X_incorrect_cnt_latest.tocsr()[idx[0], :].toarray().reshape(-1)
            stud_B_incor = X_incorrect_cnt_latest.tocsr()[idx[1], :].toarray().reshape(-1)

            #in general the x_range is number of problem-steps
            x_range = np.arange(stud_A_incor.shape[0])

            #create plots of vector pairs for cor and incor
            axarr_incor[0, j].plot(x_range, stud_A_incor, label='stud_A_incor')
            axarr_incor[1, j].plot(x_range, stud_B_incor, label='stud_B_incor')
            axarr_incor[2, j].plot(x_range, (stud_A_incor - stud_B_incor), label='stud_A_incor - stud_B_incor')

        # add a big axes, hide frame
        fig_incor.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.title('Student Vector Pair Samples for {}: Incorrect T/F'.format(title))
        plt.xlabel('{1st || 50th || 99th} percentile similarity student pairs')
        plt.ylabel('{Stud_A - Stud_B || Stud_B || Stud_A}')
        plt.show()
