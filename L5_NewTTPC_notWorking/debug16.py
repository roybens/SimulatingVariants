# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 22:06:54 2021

@author: bensr
"""
import generalized_genSim_shorten_time as ggsd
from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import optimize, stats
import yaml



def gen_sim_data(channel):
    '''
    Generate simulated data using the current NEURON state. Returns dictionary
    with activation, inactivation, tau, and recovery data.
    ---
    Return sim_data: dictionary of simulated data
    '''
    sim_data = {}

    #simulate activation
    act, act_sweeps, act_i = ggsd.activationNa12("genActivation",channel_name = channel)
    sim_data["act"] = act.to_python()
    sim_data["act sweeps"] = act_sweeps.tolist()

    #simulate inactivation
    inact, inact_sweeps,inact_i = ggsd.inactivationNa12("genInactivation",channel_name = channel)
    sim_data["inact"] = inact.to_python()
    sim_data["inact sweeps"] = inact_sweeps.tolist()

    #calculate taus from inactivation
    taus, tau_sweeps, tau0 = ggsd.find_tau_inact(inact_i)
    sim_data["taus"] = taus
    sim_data["tau sweeps"] = tau_sweeps
    sim_data["tau0"] = tau0

    #simulate recovery
    recov, recov_times = ggsd.recInactTauNa12("genRecInact",channel_name = channel)
    sim_data["recov"] = recov
    sim_data["recov times"] = recov_times
    return sim_data
def gen_figure_given_params(yaml_fn1,yaml_fn2,chan,names, save=True, file_name=None,rmse=None):
    '''
    Generate figure including all curves and tau value for a mutant.
    ---
    Param mutant: name of mutant
    Param exp: name of experiment
    Param params: list of params to use
    Param target_data: data to plot for comparison
    Param save: boolean for saving figure
    Param file_name: desired file name to save as
    '''
    #set-up figure
    plt.close()
    fig, axs = plt.subplots(2, figsize=(10,10))
    fig.suptitle("Mutant: {} \n Experiment: {}".format(names[0], names[1]))
    if yaml_fn1 is not None:
        update_16(yaml_fn1,chan)
    try:
        sim_data1 = gen_sim_data(chan)
    except ZeroDivisionError: #catch error to prevent bad individuals from halting run
        print("ZeroDivisionError when generating sim_data, returned infinity.")
        sim_data =None
        return 
    update_16(yaml_fn2,chan)
    try:
        sim_data2 = gen_sim_data(chan)
    except ZeroDivisionError: #catch error to prevent bad individuals from halting run
        print("ZeroDivisionError when generating sim_data, returned infinity.")
        sim_data =None
        return
    data = [sim_data1, sim_data2]
    max_calls = 500

    #plot inactivation and activation curves on same axis

    axs[0].set_xlabel('Voltage')
    axs[0].set_ylabel('Fraction In/activated')
    axs[0].set_title("Inactivation and Activation Curves")
    for i in range(len(data)):
        data_pts = data[i]["inact"]
        sweeps = data[i]["inact sweeps"]
        try:
            popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=max_calls)
        except:
            print("Very bad voltages in inact")
            return
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        
        if i == 0:
            axs[0].scatter(sweeps, data_pts, color='black',marker='s')
            axs[0].plot(even_xs, curve, color='black', label=names[i]+" inactivation")
        else:
            axs[0].scatter(sweeps, data_pts, color='red',marker='s')
            axs[0].plot(even_xs, curve, color='red', label=names[i]+" inactivation")
    for i in range(len(data)):
        data_pts = data[i]["act"]
        sweeps = data[i]["act sweeps"]
        try:
            popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=max_calls)
        except:
            print("Very bad voltages in act")    
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        if i == 0:
            axs[0].scatter(sweeps, data_pts, color='black',marker='o')
            axs[0].plot(even_xs, curve, color='black', label=names[i]+" activation")
        else:
            axs[0].scatter(sweeps, data_pts, color='red',marker='o')
            axs[0].plot(even_xs, curve, color='red', label=names[i]+" activation")
    axs[0].legend()

    #plot recovery curves
    axs[1].set_xlabel('Log(Time)')
    axs[1].set_ylabel('Fractional Recovery')
    axs[1].set_title("Recovery from Inactivation")
    for i in range(len(data)):
        if i == 0:
            curr_marker = 'o'
            fit_color = 'black'
        else:
            curr_marker = 's'
            fit_color = 'red'
        data_pts = data[i]["recov"]
        times = data[i]["recov times"]
        try:
            popt, pcov = optimize.curve_fit(fit_hill, times, data_pts, maxfev=max_calls)
        except:
            print("Very bad voltages in recovery")
            return
        even_xs = np.linspace(times[0], times[len(times)-1], 100)
        curve = fit_hill(even_xs, *popt)
        axs[1].plot(np.log(even_xs), curve, c=fit_color,label=names[i]+" recovery")
        axs[1].scatter(np.log(times), data_pts, label=names[i], color=fit_color,marker=curr_marker)
        
    axs[1].legend()

    #add text containing tau information
    fig.text(.5, .92, "\n simdata1 tau: {}, simdata2 tau: {}\n rmse: {}".format(sim_data1['tau0'], sim_data2['tau0'],rmse), ha='center')
    plt.show()

    #save figure
    if save:
        if file_name is None:
            file_name = "{}_{}_plots".format(names[0], names[1]).replace(" ", "_")
        fig.savefig("./curves/"+file_name+'.eps')
        fig.savefig("./curves/"+file_name+'.pdf')



def gen_curves(data, names):
    '''
    Plot inactivation, activation, and recovery curves separately for each data set.
    ---
    Param data: list of data dictionaries
    Param names: list of names for data
    '''
    #plot inactivation
    for i in range(len(data)):
        data_pts = data[i]["inact"]
        sweeps = data[i]["inact sweeps"]
        #fit sigmoid curve to data
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        plt.scatter(sweeps, data_pts)
        plt.plot(even_xs, curve, label=names[i])
    plt.legend()
    plt.xlabel('Voltage')
    plt.ylabel('Fraction Inactivated')
    plt.title("Inactivation Curve")
    plt.show()

    #plot activation
    for i in range(len(data)):
        data_pts = data[i]["act"]
        sweeps = data[i]["act sweeps"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        plt.scatter(sweeps, data_pts)
        plt.plot(even_xs, curve, label=names[i])
    plt.legend()
    plt.xlabel('Voltage')
    plt.ylabel('Fraction Activated')
    plt.title("Activation Curve")
    plt.show()

    #plot recovery
    for i in range(len(data)):
        data_pts = data[i]["recov"]
        times = data[i]["recov times"]
#        popt, pcov = optimize.curve_fit(fit_hill, times, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
#        even_xs = np.linspace(times[0], times[len(sweeps)-1], 100)
#        curve = fit_hill(even_xs, *popt)
        plt.scatter(np.log(times), data_pts, label=names[i])
 #       plt.plot(even_xs, curve, label=names[i])

    plt.legend()
    plt.xlabel('Log(Time)')
    plt.ylabel('Fractional Recovery')
    plt.title("Recovery from Inactivation")
    plt.show()
    
    
def update_16(yaml_fn,mod_suffix):
    with open(yaml_fn, 'r') as stream:
        param_dict = yaml.safe_load(stream)
    p_names = list(param_dict.keys())
    for p in p_names:
        p_value = param_dict[p]
        if ('Ena' in p):
            p='Ena'
            continue
        str = f'{p}_{mod_suffix}={p_value}'
        
        h(str)
def fit_sigmoid(x, a, b):
    '''
    Fit a sigmoid curve to the array of datapoints.
    '''
    return 1.0 / (1.0+np.exp(-a*(x-b)))

def fit_exp(x, a, b, c):
    '''
    Fit an exponential curve to an array of datapoints.
    '''
    return a*np.exp(-b*x)+c
def fit_dblexp(x, a, b, c, d):
    return a * np.exp(-b * x) + c * np.exp(-d * x)
def fit_hill(x, a, b, c): # Hill sigmoidal equation from zunzun.com
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b)) 

gen_figure_given_params(None,'../SCN8A_WT_opt_params.yaml','na16mut',['orig','fitted_na16'])
# na16_origsim = gen_sim_data('na16mut')
# update_16('../SCN8A_WT_opt_params.yaml','na16mut')
# na16_modified = gen_sim_data('na16mut')
# gen_curves([na16_origsim,na16_modified],['na16','na16_opt'])