###########################3
## Functions for fitting sigmoid curves to activation and inactivation as well as
## 2 phase inactivation for recovery.
## Author: Michael Lam, adapted from code in optimize_na_ga_v2

import matplotlib.pyplot as plt
import numpy as np
import eval_helper as eh
import generalized_genSim_shorten_time as ggsd
import generalized_genSim_shorten_time_HMM as ggsdHMM
from scipy import optimize, stats

def boltzmann(x, slope, v_half, top, bottom):
    '''
    Fit a sigmoid curve to the array of datapoints.
    '''
    return bottom +  ((top - bottom) / (1.0 + np.exp((v_half - x)/slope)))

def two_phase(x, y0, plateau, percent_fast, k_fast, k_slow):
    '''
    Fit a two-phase association curve to an array of data points X. 
    For info about the parameters, visit 
    https://www.graphpad.com/guides/prism/latest/curve-fitting/REG_Exponential_association_2phase.htm
    '''
    span_fast = (plateau - y0) * percent_fast * 0.01
    span_slow = (plateau - y0) * (100 - percent_fast) * 0.01
    return y0 + span_fast * (1 - np.exp(-k_fast * x)) + span_slow * (1 - np.exp(-k_slow * x))

def one_phase(x, y0, plateau, k):
    '''
    Fit a one-phase association curve to an array of data points X. 
    For info about the parameters, visit 
    https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_exponential_association.htm    
    '''
    return y0 + (plateau - y0) * (1 - np.exp(-k * x))

def calc_act_obj(channel_name, is_HMM=False):
    try:
        if not is_HMM:
            gnorm_vec, v_vec, all_is = ggsd.Activation(channel_name=channel_name, step=5).genActivation()
        else:
            gnorm_vec, v_vec, all_is = ggsdHMM.Activation(channel_name=channel_name, step=5).genActivation()
    except:
        print('Couldn\'t generate activation data')
        return (1000, 1000, 1000, 1000)
    try:
        popt, pcov = optimize.curve_fit(boltzmann, v_vec, gnorm_vec)
    except:
        print("Couldn't fit curve to activation.")
        return (1000, 1000, 1000, 1000)
    gv_slope, v_half, top, bottom = popt
    return gv_slope, v_half, top, bottom


def calc_inact_obj(channel_name, is_HMM=False):
    try:
        if not is_HMM:
            inact = ggsd.Inactivation(channel_name=channel_name, step=5)
            inorm_vec, v_vec, all_is = inact.genInactivation()
        else:
            inact = ggsdHMM.Inactivation(channel_name=channel_name, step=5)
            inorm_vec, v_vec, all_is = inact.genInactivation()
    except:
        print('Couldn\'t generate inactivation data')
        return (1000, 1000, 1000, 1000, 1000)
    try:
        popt, pcov = optimize.curve_fit(boltzmann, v_vec, inorm_vec)
    except:
        print("Couldn't fit curve to inactivation.")
        return (1000, 1000, 1000, 1000, 1000)
    ssi_slope, v_half, top, bottom = popt
    tau0 = inact.get_just_tau0()
    # taus, tau_sweeps, tau0 = ggsd.find_tau_inact(all_is)
    return ssi_slope, v_half, top, bottom, tau0

def calc_recov_obj(channel_name, is_HMM=False):
    try:
        if not is_HMM:
            rec_inact_tau_vec, recov_curves, times = ggsd.RFI(channel_name=channel_name).genRecInactTau()
        else:
            rec_inact_tau_vec, recov_curves, times = ggsdHMM.RFI(channel_name=channel_name).genRecInactTau()
    except:
        print('Couldn\'t generate recovery data')
        return (1000, 1000, 1000, 1000, 1000)
    recov_curve = recov_curves[0]
    try:
        popt, pcov = optimize.curve_fit(two_phase, times, recov_curve)
    except:
        print("Couldn't fit curve to recovery.")
        #return (1000, 1000, 1000, 1000, 1000, 1000)
        return (1000, 1000, 1000, 1000, 1000)

    y0, plateau, percent_fast, k_fast, k_slow = popt
    #tau0 = rec_inact_tau_vec[0]
    #return y0, plateau, percent_fast, k_fast, k_slow, tau0 
    return y0, plateau, percent_fast, k_fast, k_slow

