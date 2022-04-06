import eval_helper as eh
'''
This file contains various scoring functions that can be used in the evaluator's
calc_rmse function. 
'''

def calc_rmse_standard(target_data):
    '''
    Calculates the inactivation, activation, recovery, and tau0 rmse
    in that order compared to the current state of the neuron.

    params: 
        target_data: a dictionary of the experimental data in standard format
    returns:
        List of rmse values in order
    '''
    try:
        sim_data = eh.gen_sim_data()
    except ZeroDivisionError: #catch error to prevent bad individuals from halting run
        print("ZeroDivisionError when generating sim_data, returned infinity.")
        sim_data =None
        return (1000,1000,1000,1000)
    try:
        inds = target_data["inact sig inds"]
        squared_diffs = [(target_data['inact'][i]-sim_data['inact'][i])**2 for i in inds]
        inact_rmse = (sum(squared_diffs)/len(inds))**.5

        inds = target_data["act sig inds"]
        squared_diffs = [(target_data['act'][i]-sim_data['act'][i])**2 for i in inds]
        act_rmse = (sum(squared_diffs)/len(inds))**.5

        inds = target_data["recov sig inds"]
        squared_diffs = [(target_data['recov'][i]-sim_data['recov'][i])**2 for i in inds]
        recov_rmse = (sum(squared_diffs)/len(inds))**.5

        tau_rmse = ((target_data['tau0']-sim_data['tau0'])**2)**.5
    except OverflowError:
        print('OverflowError when calculating rmse, returned infinity.')
        return (1000, 1000, 1000, 1000)
    return [inact_rmse, act_rmse, recov_rmse, tau_rmse]

def calc_rmse_sans_tau(target_data):
    '''
    Calculates the inactivation, activation, and recovery rmse
    in that order

    params:
        target_data: a dictionary of the experimental data in standard format
    returns:
        List of rmse values in order
    '''
    try:
        sim_data = eh.gen_sim_data()
    except ZeroDivisionError: #catch error to prevent bad individuals from halting run
        print("ZeroDivisionError when generating sim_data, returned infinity.")
        sim_data =None
        return (1000,1000,1000)
    try:
        inds = target_data["inact sig inds"]
        squared_diffs = [(target_data['inact'][i]-sim_data['inact'][i])**2 for i in inds]
        inact_rmse = (sum(squared_diffs)/len(inds))**.5

        inds = target_data["act sig inds"]
        squared_diffs = [(target_data['act'][i]-sim_data['act'][i])**2 for i in inds]
        act_rmse = (sum(squared_diffs)/len(inds))**.5

        inds = target_data["recov sig inds"]
        squared_diffs = [(target_data['recov'][i]-sim_data['recov'][i])**2 for i in inds]
        recov_rmse = (sum(squared_diffs)/len(inds))**.5

    except OverflowError:
        print('OverflowError when calculating rmse, returned infinity.')
        return (1000, 1000, 1000)
    return [inact_rmse, act_rmse, recov_rmse]

