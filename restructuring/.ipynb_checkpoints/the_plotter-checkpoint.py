
   
import generalized_genSim_shorten_time as ggsd
import generalized_genSim_shorten_time_HMM as ggsdHMM
import matplotlib.pyplot as plt
import curve_fitting as cf
from scipy import optimize
from eval_helper_na12 import set_param_currh, wt_params, find_peak_amp, read_mutant_protocols
import os
import json
import numpy as np
import matplotlib.backends.backend_pdf
global currh

def set_channel(channel_suffix = 'na12'):
    global currh
    currh = ggsd.Activation(channel_name = channel_suffix).h
    
 
def find_persistent_current():
    """
    returns the persistent current, gieven that the NEURON model already has parameters properly set
    """
    
    ramp = ggsd.Ramp(channel_name = 'na12')
    ramp.genRamp()
    return ramp.persistentCurrent()


def plotInactivation_TCurrDensityRelation(inact_obj, plt,color):
    [plt.plot(inact_obj.t_vec[-800:-700], inact_obj.all_is[i][-800:-700], c=color) for i in np.arange(inact_obj.L)]
        
        
def plotInactivation_Tau_0mV_plt(inact_obj, plt,color, upper = 700):

    diff = 0
    if color == 'red':
        diff = 1.5

    def fit_expon(x, a, b, c):
        return a + b * np.exp(-1 * c * x)
    act = ggsd.Activation(channel_name = 'na12')
    act.clamp_at_volt(0)
    starting_index = list(act.i_vec).index(act.find_ipeaks_with_index()[1])

    t_vecc = act.t_vec[starting_index:upper]
    i_vecc = act.i_vec[starting_index:upper]
    popt, pcov = optimize.curve_fit(fit_expon,t_vecc,i_vecc, method = 'dogbox')

    tau = 1/popt[2]
    #tau = 1000 * tau
    xmid = (max(t_vecc) + min(t_vecc))/2
    ymid = (max(i_vecc) + min(i_vecc))/2
    if color == 'red':
        diff = ymid*0.2
    fitted_i = fit_expon(act.t_vec[starting_index:upper],popt[0],popt[1],popt[2])
    plt.plot(act.t_vec[starting_index:upper], fitted_i, c=color)
    plt.plot(t_vecc,i_vecc,'o',c=color)
    plt.text(xmid, ymid + diff, f"Tau at 0 mV: {tau}", color=color)

    return tau


    # select 0 mV
    volt = 0  # mV
    mask = np.where(inact_obj.v_vec == volt)[0]
    curr = np.array(inact_obj.all_is)[mask][0]
    time = np.array(inact_obj.t_vec)[1:]
    # fit exp: IFit(t) = A * exp (-t/τ) + C
    ts, data, xs, ys, tau = inact_obj.find_tau0_inact(curr)
    # plot
    plt.plot(ts, data, color=color)
    plt.plot(xs, ys, color=color)
    formatted_tau0 = np.round(tau, decimals=3)

    return tau

    



def plotInactivation_TimeVRelation_plt(inact_obj, plt, color):
    [plt.plot(inact_obj.t_vec, inact_obj.all_v_vec_t[i], c=color) for i in np.arange(inact_obj.L)]

def plotInactivation_VInormRelation_plt(inact_obj, plt, color):

    diff = 0
    if color == 'red':
        diff = 0.5
    plt.plot(inact_obj.v_vec, inact_obj.inorm_vec, 'o', c=color)
    ssi_slope, v_half, top, bottom, tau0 = cf.calc_inact_obj(inact_obj.channel_name)
    formatted_ssi_slope = np.round(ssi_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5 + diff, f'Slope: {1/formatted_ssi_slope}', c = color)
    plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c = color)
    x_values_v = np.arange(inact_obj.st_cl, inact_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, ssi_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c=color)
    return (formatted_v_half, formatted_ssi_slope)

def plotActivation_TimeVRelation_plt(act_obj,plt,color):
    [plt.plot(act_obj.t_vec, act_obj.all_v_vec_t[i], c=color) for i in np.arange(act_obj.L)]
    
    
def plotActivation_TCurrDensityRelation_plt(act_obj,plt,color):
    curr = np.array(act_obj.all_is)
    mask = np.where(np.logical_or(act_obj.v_vec == 0, act_obj.v_vec == 0))
    [plt.plot(act_obj.t_vec[190:300], curr[i][190:300], c=color) for i in np.arange(len(curr))[mask]]



def plotActivation_VGnorm_plt(act_obj,plt,color):
    """
    Saves activation plot as PGN file.
    """

    diff = 0
    if color == 'red':
        diff = 0.5 

    plt.plot(act_obj.v_vec, act_obj.gnorm_vec, 'o', c=color)
    gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj.channel_name)
    formatted_gv_slope = np.round(gv_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5 + diff, f'Slope: {1/formatted_gv_slope}', c = color)
    plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c = color)
    x_values_v = np.arange(act_obj.st_cl, act_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c=color)
    return (formatted_v_half, formatted_gv_slope)

def plotActivation_IVCurve_plt(act_obj,plt,color):

    plt.plot(np.array(act_obj.v_vec), np.array(act_obj.ipeak_vec), 'o', c=color)
    #plt.text(-110, -0.05, 'Vrev at ' + str(round(act_obj.vrev, 1)) + ' mV', fontsize=10, c='blue')
    formatted_peak_i = np.round(min(act_obj.ipeak_vec), decimals=2)
    #plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps

def make_act_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt = wt_params, filename = 'jinan_plots_out.pdf'):
    """
    input:  
        new_params: a set of variant parameters 
        param_values_wt: WT parameters. Defaulted to NA 16 WT.
        filename: name of the pdf file into which we want to store the figures
    returns:
        no return values, it just makes the activation plots for each set of parameters
    """
    
    set_param = set_param_currh
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []
    
    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized conductance')
    plt.title(f'Activation: {mutant_name}')

    set_param(currh, param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na12')
    wt_act.genActivation()
    # (formatted_v_half, formatted_gv_slope)
    act_v_half_wt, act_slope_wt = plotActivation_VGnorm_plt(wt_act, plt, 'black')
    
    act_slope_wt = 1/act_slope_wt

    set_param(currh, new_params)
    mut_act = ggsd.Activation(channel_name = 'na12')
    mut_act.genActivation()
    act_v_half_mut, act_slope_mut = plotActivation_VGnorm_plt(mut_act, plt, 'red')
    act_slope_mut = 1/act_slope_mut
    

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title(f'Activation: {mutant_name} IV Curve')

    set_param(currh, param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na12')
    wt_act.genActivation()
    plotActivation_IVCurve_plt(wt_act, plt, 'black')

    set_param(currh, new_params)
    mut_act = ggsd.Activation(channel_name = 'na12')
    mut_act.genActivation()
    plotActivation_IVCurve_plt(mut_act, plt, 'red')

    ############################################################################################################
    # figures.append(plt.figure())
    # plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(mV)$')
    # plt.title('Activation Time/Voltage relation')

    # set_param(param_values_wt)
    # wt_act = ggsd.Activation(channel_name = 'na12')
    # wt_act.genActivation()
    # wt_act.plotActivation_TimeVRelation_plt(plt, 'black')

    # set_param(new_params)
    # mut_act = ggsd.Activation(channel_name = 'na12')
    # mut_act.genActivation()
    # mut_act.plotActivation_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('I $(mA/cm^2)$')
    plt.title(f'Activation waveform at 0mV: {mutant_name}')

    set_param(currh, param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na12')
    wt_act.genActivation()
    plotActivation_TCurrDensityRelation_plt(wt_act, plt, 'black')

    set_param(currh, new_params)
    mut_act = ggsd.Activation(channel_name = 'na12')
    mut_act.genActivation()
    plotActivation_TCurrDensityRelation_plt(mut_act, plt, 'red')
    
    
 ############################################################################################################

    set_param(currh, param_values_wt)
    wt_peak_amp = find_peak_amp()

    set_param(currh, new_params)
    mut_peak_amp = find_peak_amp()
    
    
############################################################################################################    
    peak_amp_dict = {"T400RAdult": 0.645, "I1640NAdult": 0.24, "m1770LAdult": 0.4314, "neoWT": 0.748, "T400RAneo": 0.932, "I1640NNeo": 0.28, "m1770LNeo": 1}
    
    figures.append(plt.figure())
    goal_dict = read_mutant_protocols(mutant_protocol_csv_name, mutant_name)
    plt.text(0.4,0.9,"(actual, goal)")
    plt.text(0.1,0.7,"activation v half: " + str((act_v_half_mut - act_v_half_wt , goal_dict['dv_half_act'])))
    plt.text(0.1,0.5,"activation slope: " + str((act_slope_mut/act_slope_wt , goal_dict['gv_slope']/100)))
    plt.text(0.1,0.3,"peak amp: " + str((mut_peak_amp/wt_peak_amp , peak_amp_dict[mutant_name])))

    

    plt.axis('off')
    for fig in figures: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()

    ############################################################################################################
    
    print("(actual, goal)")
    print("activation v half: " + str((act_v_half_mut - act_v_half_wt , goal_dict['dv_half_act'])))
    print("activation slope: " + str((act_slope_mut/act_slope_wt , goal_dict['gv_slope']/100)))

    

def make_inact_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt = wt_params, filename = 'jinan_plots_out.pdf'):
    """
     input:  
        new_params: a set of variant parameters 
        param_values_wt: WT parameters. Defaulted to NA 16 WT.
        filename: name of the pdf file into which we want to store the figures
    returns:
        no return values, it just makes the inactivation plots for each set of parameters
    """
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []
    
    set_param = set_param_currh
    
    ############################################################################################################

    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized current')
    plt.title(f'Inactivation: {mutant_name}')

    set_param(currh, param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na12')
    wt_inact.genInactivation()
    inact_v_half_wt, inact_slope_wt = plotInactivation_VInormRelation_plt(wt_inact, plt, 'black')
    inact_slope_wt = 1/inact_slope_wt

    set_param(currh, new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na12')
    mut_inact.genInactivation()
    inact_v_half_mut, inact_slope_mut = plotInactivation_VInormRelation_plt(mut_inact, plt, 'red')
    inact_slope_mut = 1/inact_slope_mut


    ############################################################################################################
    # figures.append(plt.figure())
    # plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(mV)$')
    # plt.title('Inactivation Time/Voltage relation')

    # set_param(param_values_wt)
    # wt_inact = ggsd.Inactivation(channel_name = 'na12')
    # wt_inact.genInactivation()
    # wt_inact.plotInactivation_TimeVRelation_plt(plt, 'black')

    # set_param(new_params)
    # mut_inact = ggsd.Inactivation(channel_name = 'na12')
    # mut_inact.genInactivation()
    # mut_inact.plotInactivation_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title(f'Inactivation: {mutant_name}')

    set_param(currh, param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na12')
    wt_inact.genInactivation()
    plotInactivation_TCurrDensityRelation(wt_inact, plt, 'black')

    set_param(currh, new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na12')
    mut_inact.genInactivation()
    plotInactivation_TCurrDensityRelation(mut_inact, plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title(f'Inactivation Tau at 0 mV: {mutant_name}')

    set_param(currh, param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na12')
    wt_inact.genInactivation()
    wt_tau = plotInactivation_Tau_0mV_plt(wt_inact, plt, 'black')
    wt_per_cur = find_persistent_current()[0]

    set_param(currh, new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na12')
    mut_inact.genInactivation()
    mut_tau = plotInactivation_Tau_0mV_plt(mut_inact, plt, 'red')
    mut_per_cur = find_persistent_current()[0]

    
    figures.append(plt.figure())
    goal_dict = read_mutant_protocols(mutant_protocol_csv_name, mutant_name)
    plt.text(0.4,0.9,"(actual, goal)")
    plt.text(0.1,0.7,"tau: " + str((mut_tau/wt_tau , goal_dict['tau0']/100)))
    plt.text(0.1,0.5,"persistent current: " + str((mut_per_cur/wt_per_cur, goal_dict['persistent']/100)))
    plt.text(0.1,0.3,"inactivation v half: " + str((inact_v_half_mut - inact_v_half_wt , goal_dict['dv_half_ssi'])))
    plt.text(0.1,0.1,"inactivation slope: " + str((inact_slope_mut/inact_slope_wt , goal_dict['ssi_slope']/100)))
    plt.axis('off')
    for fig in figures: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()