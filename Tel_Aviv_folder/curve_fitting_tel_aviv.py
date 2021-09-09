###########################3
## Functions for fitting sigmoid curves to activation and inactivation as well as
## 2 phase inactivation for recovery.
## Authors: Michael Lam, Chastin Chung
##adapted from code in optimize_na_ga_v2

import matplotlib.pyplot as plt
import numpy as np
import eval_helper as eh
import generalized_genSim_tel_aviv as ggsd
from scipy import optimize, stats

def boltzmann(x, slope, v_half, top, bottom):
    '''
    Fit a sigmoid curve to the array of datapoints.
    '''
    return bottom + ((top - bottom) / (1.0 + np.exp((v_half - x)/slope)))

def one_phase(x, y0, plateau, k):
    '''
    Fit a one-phase association curve to an array of data points X. 
    For info about the parameters, visit 
    https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_exponential_association.htm    
    '''
    return y0 + (plateau - y0) * (1 - np.exp(-k * x))

def gen_figure_given_params(params, save=True, file_name=None,mutant='N_A', exp='N_A',rmse=None, plot=False):
    #set-up figure
    eh.change_params(params, scaled=False)
    param_dict = {}
    plt.close()
    fig, axs = plt.subplots(3, figsize=(10,10))
    fig.suptitle("Mutant: {} \n Experiment: {}".format(mutant, exp))
        
    # Inactivation curve
    inorm_vec, v_vec, all_is = ggsd.Inactivation().genInactivation()
    inorm_array = np.array(inorm_vec)
    v_array = np.array(v_vec)
    ssi_slope, v_half, top, bottom = calc_inact_obj()
    even_xs = np.linspace(v_array[0], v_array[len(v_array)-1], 100)
    curve = boltzmann(even_xs, ssi_slope, v_half, top, bottom)
    axs[0].set_xlabel('Voltage (mV)')
    axs[0].set_ylabel('Fraction Inactivated')
    axs[0].set_title("Inactivation Curve")
    axs[0].scatter(v_array, inorm_array, color='black',marker='s')
    axs[0].plot(even_xs, curve, color='red', label="Inactivation")
    axs[0].text(-10, 0.5, 'Slope: ' + str(ssi_slope) + ' /mV')
    axs[0].text(-10, 0.3, 'V50: ' + str(v_half) + ' mV')
    axs[0].legend()

    # Activation curve
    gnorm_vec, v_vec, all_is = ggsd.Activation().genActivation()
    gnorm_array = np.array(gnorm_vec)
    v_array = np.array(v_vec)
    gv_slope, v_half, top, bottom = calc_act_obj()
    even_xs = np.linspace(v_array[0], v_array[len(v_array)-1], 100)
    curve = boltzmann(even_xs, gv_slope, v_half, top, bottom)
    axs[1].set_xlabel('Voltage (mV)')
    axs[1].set_ylabel('Fraction Activated')
    axs[1].set_title("Activation Curve")
    axs[1].scatter(v_array, gnorm_array, color='black',marker='s')
    axs[1].plot(even_xs, curve, color='red', label="Activation")
    axs[1].text(-10, 0.5, 'Slope: ' + str(gv_slope) + ' /mV')
    axs[1].text(-10, 0.3, 'V50: ' + str(v_half) + ' mV')
    axs[1].legend()
        
    #Recovery Curve
    rec_inact_tau_vec, recov_curves, times = ggsd.RFI().genRecInactTau()
    times = np.array(times)
    data_pts = np.array(recov_curves[0])
    axs[2].set_xlabel('Log(Time)')
    axs[2].set_ylabel('Fractional Recovery')
    axs[2].set_title("Recovery from Inactivation")
    even_xs = np.linspace(times[0], times[len(times)-1], 100)
    y0, plateau, k, tau  = calc_recov_obj()
    curve = one_phase(even_xs, y0, plateau, k)
    axs[2].plot(np.log(even_xs), curve, c='red',label="Recovery Fit")
    axs[2].scatter(np.log(times), data_pts, label='Recovery', color='black')
    plt.show()

def calc_act_obj(channel_name):
    try:
        gnorm_vec, v_vec, all_is = ggsd.Activation(channel_name=channel_name).genActivation()
    except:
        print('Couldn\'t generate activation data')
        return (1000, 1000, 1000, 1000)
    try:
        popt, pcov = optimize.curve_fit(boltzmann, v_vec, gnorm_vec)
    except:
        print("Very bad voltages in activation.")
        return (1000, 1000, 1000, 1000)
    gv_slope, v_half, top, bottom = popt
    return gv_slope, v_half, top, bottom


def calc_inact_obj(channel_name):
    try:
        inorm_vec, v_vec, all_is = ggsd.Inactivation(channel_name=channel_name).genInactivation()
    except:
        print('Couldn\'t generate inactivation data')
        return (1000, 1000, 1000, 1000)
    try:
        popt, pcov = optimize.curve_fit(boltzmann, v_vec, inorm_vec)
    except:
        print("Very bad voltages in inactivation.")
        return (1000, 1000, 1000, 1000)
    ssi_slope, v_half, top, bottom = popt
    return ssi_slope, v_half, top, bottom

def calc_recov_obj(channel_name):
    try:
        rec_inact_tau_vec, recov_curves, times = ggsd.RFI(channel_name=channel_name).genRecInactTau()
    except:
        print('Couldn\'t generate recovery data')
        return (1000, 1000, 1000, 1000)
    recov_curve = recov_curves[0]
    try:
        popt, pcov = optimize.curve_fit(one_phase, times, recov_curve)
    except:
        print("Very bad voltages in Recovery.")
        return (1000, 1000, 1000, 1000)

    y0, plateau, k = popt
    tau = 1/k
    return y0, plateau, k, tau 


