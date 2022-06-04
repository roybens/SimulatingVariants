###########################3
## Functions for fitting sigmoid curves to activation and inactivation as well as
## 2 phase inactivation for recovery.
## Author: Michael Lam, adapted from code in optimize_na_ga_v2

import numpy as np
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

def calc_act_obj(act_obj):
    try:
        # import ipdb
        # ipdb.set_trace()
        gnorm_vec, v_vec, all_is = act_obj.genActivation()
    except:
        #print('Couldn\'t generate activation data')
        return (1000, 1000, 1000, 1000)
    try:
        popt, pcov = optimize.curve_fit(boltzmann, v_vec, gnorm_vec)
    except:
        #print("Couldn't fit curve to activation.")
        return (1000, 1000, 1000, 1000)
    gv_slope, v_half, top, bottom = popt
    return gv_slope, v_half, top, bottom

def get_tau0mv(act_obj):
    act_obj.get_Tau_0mV(act_obj)
    return tau
def calc_inact_obj(inact_obj):
    try:
        inorm_vec, v_vec, all_is = inact_obj.genInactivation()
    except:
        #print('Couldn\'t generate inactivation data')
        return (1000, 1000, 1000, 1000)
    try:
        popt, pcov = optimize.curve_fit(boltzmann, v_vec, inorm_vec)
    except:
        #print("Couldn't fit curve to inactivation.")
        return (1000, 1000, 1000, 1000)
    ssi_slope, v_half, top, bottom = popt
    # taus, tau_sweeps, tau0 = ggsd.find_tau_inact(all_is)
    return ssi_slope, v_half, top, bottom


def calc_recov_obj(recov_obj):
    try:
        rec_inact_tau_vec, recov_curves, times = recov_obj.genRecInactTau()
    except:
        #print('Couldn\'t generate recovery data')
        return (1000, 1000, 1000, 1000, 1000)
    recov_curve = recov_obj.rec_vec
    try:
        popt, pcov = optimize.curve_fit(two_phase, np.log(times), recov_curve)
    except:
        #print("Couldn't fit curve to recovery.")
        return (1000, 1000, 1000, 1000, 1000)

    y0, plateau, percent_fast, k_fast, k_slow = popt
    #tau0 = rec_inact_tau_vec[0]
    #return y0, plateau, percent_fast, k_fast, k_slow, tau0 
    return y0, plateau, percent_fast, k_fast, k_slow

# Technically not fitting any curves here, but Michael is placing this here for consistency until a better
# place is found.
def calc_tau0_obj(act_obj, is_HMM=False):
    try:
        tau0 = act_obj.get_Tau_0mV()
    except:
        #print('Couldn\'t generate tau0 data')
        return 1000

# Technically not fitting any curves here, but Michael is placing this here for consistency until a better
# place is found.
def calc_peak_amp_obj(act_obj, is_HMM=False):
    try:
        peak_amp = act_obj.find_peak_amp()
        return peak_amp
    except:
 #       print('Couldn\'t generate peak_amp data')
        return 1000
    
# Technically not fitting any curves here, but Michael is placing this here for consistency until a better
# place is found.
def calc_time_to_peak_obj(act_obj, is_HMM=False):
    try:
        ttp = act_obj.find_time_to_peak()
        return ttp
    except:
  #      print('Couldn\'t generate time-to-peak data')
        return 1000
# Technically not fitting any curves here, but Michael is placing this here for consistency until a better
# place is found.
def calc_act_prst_curr(act_obj, v_trace = 0,is_HMM=False):
    try:
        prst_curr,t_mask = act_obj.get_perst_curr(v_trace)
        return np.mean(prst_curr)
    except Exception as e:
        print(e)
        return 1000