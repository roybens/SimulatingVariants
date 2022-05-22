import os

import generalized_genSim_shorten_time_HMM as ggsdHMM
import generalized_genSim_shorten_time as ggsd
import matplotlib.backends.backend_pdf
import eval_helper as eh
import matplotlib.pyplot as plt
import curve_fitting as cf
import numpy as np
from scipy import optimize

def set_param(param, is_HMM):
    eh.change_params(param, scaled=False, is_HMM=is_HMM)
def read_peak_amp_dict():
    return {"T400RAdult": 0.645, "I1640NAdult": 0.24, "m1770LAdult": 0.4314, "neoWT": 0.748, "T400RAneo": 0.932, "I1640NNeo": 0.28, "m1770LNeo": 1, "K1260E" : 1}
        
def read_mutant_protocols(mutant_protocols_csv, mutant):
    '''
    Reads data for a single MUTANT from a csv of mutant protocols.
    Returns a dictionary with all the relevant protocols for that 
    MUTANT.
    '''
    lines = []
    with open(mutant_protocols_csv, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]

    #Each line[0] except the first should contain the name of the mutant 
    mutant_line = []
    for line in lines:
        if line[0] == mutant:
            mutant_line = line
            break
    if mutant_line == []:
        raise NameError('Invalid mutant name, or mutant is not yet in CSV database')
    protocols_dict = {}
    protocols_dict['dv_half_act'] = float(mutant_line[1])
    protocols_dict['gv_slope'] = float(mutant_line[2])
    protocols_dict['dv_half_ssi'] = float(mutant_line[3])
    protocols_dict['ssi_slope'] = float(mutant_line[4])
    protocols_dict['tau_fast'] = float(mutant_line[5])
    protocols_dict['tau_slow'] = float(mutant_line[6])
    protocols_dict['percent_fast'] = float(mutant_line[7])
    protocols_dict['udb20'] = float(mutant_line[8])
    protocols_dict['tau0'] = float(mutant_line[9])
    protocols_dict['ramp'] = float(mutant_line[10])
    protocols_dict['persistent'] = float(mutant_line[11])

    return protocols_dict

def find_persistent_current(is_HMM):
    """
    returns the persistent current, gieven that the NEURON model already has parameters properly set
    """
    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd
        
    ramp = module_name.Ramp()
    ramp.genRamp()
    return ramp.persistentCurrent()

def find_peak_amp(channel_name, is_HMM):
    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd
        
    act = module_name.Activation(channel_name = channel_name)
    act.clamp_at_volt(0)
    return act.ipeak_vec[0]

def plotActivation_VGnorm(act_obj):
    """
    Saves activation plot as PGN file.
    """
    plt.figure()
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized conductance')
    plt.title('Activation: Voltage/Normalized conductance')
    plt.plot(act_obj.v_vec, act_obj.gnorm_vec, 'o', c='black')
    gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj)
    formatted_gv_slope = np.round(gv_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5, f'Slope: {formatted_gv_slope}')
    plt.text(-10, 0.3, f'V50: {formatted_v_half}')
    x_values_v = np.arange(act_obj.st_cl, act_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c='red')
    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0],
                     'Plots_Folder/HMM_Activation Voltage-Normalized Conductance Relation'))

def plotActivation_IVCurve(act_obj):
    plt.figure()
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title("Activation: IV Curve")
    plt.plot(act_obj.v_vec, act_obj.ipeak_vec, 'o', c='black')
    plt.text(-110, -0.05, 'Vrev at ' + str(round(act_obj.vrev, 1)) + ' mV', fontsize=10, c='blue')
    formatted_peak_i = np.round(min(act_obj.ipeak_vec), decimals=2)
    plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps
    # save as PGN file
    plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Activation IV Curve"))

def plotActivation_IVCurve_plt(act_obj,plt,color):

    plt.plot(np.array(act_obj.v_vec), np.array(act_obj.ipeak_vec), 'o', c=color)
    #plt.text(-110, -0.05, 'Vrev at ' + str(round(act_obj.vrev, 1)) + ' mV', fontsize=10, c='blue')
    formatted_peak_i = np.round(min(act_obj.ipeak_vec), decimals=2)
    #plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps


def plotActivation_TimeVRelation(act_obj):
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title('Activation Time/Voltage relation')
    [plt.plot(act_obj.t_vec, act_obj.all_v_vec_t[i], c='black') for i in np.arange(act_obj.L)]
    # save as PGN file
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/HMM_Activation Time Voltage Relation'))

def plotActivation_TCurrDensityRelation(act_obj,xlim = None):
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Activation Time/Current density relation')
    curr = np.array(act_obj.all_is)
    [plt.plot(act_obj.t_vec[1:], curr[i], c='black') for i in np.arange(len(curr))]
    if xlim is not None:
        for i in np.arange(len(curr)):
            plt.plot(act_obj.t_vec[1:], curr[i], c='black')
            plt.xlim(xlim)

    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Activation Time Current Density Relation"))
        
def plotActivation_TCurrDensityRelation_plt(act_obj,plt,color):
    curr = np.array(act_obj.all_is)
    mask = np.where(np.logical_or(act_obj.v_vec == 0, act_obj.v_vec == 10))
    [plt.plot(act_obj.t_vec[190:300], curr[i][190:300], c=color) for i in np.arange(len(curr))[mask]]


def plotActivation_VGnorm_plt(act_obj,plt,color):
    """
    Saves activation plot as PGN file.
    """

    diff = 0
    if color == 'red':
        diff = 0.5 


    plt.plot(act_obj.v_vec, act_obj.gnorm_vec, 'o', c=color)
    gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj)
    #gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj.channel_name, True)
    formatted_gv_slope = np.round(gv_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5 + diff, f'Slope: {formatted_gv_slope}', c = color)
    plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c = color)
    x_values_v = np.arange(act_obj.st_cl, act_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c=color)
    return (formatted_v_half, formatted_gv_slope)



def plot_act(wild_params, wild_channel_name, wild_is_HMM, mut_params, mut_channel_name, mut_is_HMM, outfile, mutant_name):
    if wild_is_HMM:
        module_name_wild = ggsdHMM
    else:
        module_name_wild = ggsd
    if mut_is_HMM:
        module_name_mut = ggsdHMM
    else:
        module_name_mut = ggsd
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)
    figures = []
    
    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized conductance')
    plt.title(f'Activation: {mutant_name}')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_act = module_name_wild.Activation(channel_name=wild_channel_name)
    wt_act.genActivation()
    # (formatted_v_half, formatted_gv_slope)
    act_v_half_wt, act_slope_wt = plotActivation_VGnorm_plt(wt_act, plt, 'black')

    set_param(mut_params, mut_is_HMM)
    mut_act = module_name_mut.Activation(channel_name=mut_channel_name)
    mut_act.genActivation()
    act_v_half_mut, act_slope_mut = plotActivation_VGnorm_plt(wt_act, plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title(f'Activation: {mutant_name} IV Curve')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_act = module_name_wild.Activation(channel_name=wild_channel_name)
    wt_act.genActivation()
    plotActivation_IVCurve_plt(wt_act, plt, 'black')

    set_param(mut_params, mut_is_HMM)
    mut_act = module_name_mut.Activation(channel_name=mut_channel_name)
    mut_act.genActivation()
    plotActivation_IVCurve_plt(mut_act, plt, 'red')

    ############################################################################################################
    # figures.append(plt.figure())
    # plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(mV)$')
    # plt.title('Activation Time/Voltage relation')

    # set_param(param_values_wt, is_HMM)
    # wt_act = module_name.Activation(channel_name = channel_name)
    # wt_act.genActivation()
    # wt_act.plotActivation_TimeVRelation_plt(plt, 'black')

    # set_param(new_params, is_HMM)
    # mut_act = module_name.Activation(channel_name = channel_name)
    # mut_act.genActivation()
    # mut_act.plotActivation_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('I $(mA/cm^2)$')
    plt.title(f'Activation waveform at 0mV: {mutant_name}')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_act = module_name_wild.Activation(channel_name=wild_channel_name)
    wt_act.genActivation()
    plotActivation_TCurrDensityRelation_plt(wt_act, plt, 'black')

    set_param(mut_params, mut_is_HMM)
    mut_act = module_name_mut.Activation(channel_name=mut_channel_name)
    mut_act.genActivation()
    plotActivation_TCurrDensityRelation_plt(mut_act, plt, 'red')
    
    
 ############################################################################################################
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_peak_amp = find_peak_amp(wild_channel_name, wild_is_HMM)

    set_param(mut_params, mut_is_HMM)
    mut_peak_amp = find_peak_amp(mut_channel_name, mut_is_HMM)
    
    
############################################################################################################    
    for fig in figures: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()

    ############################################################################################################

def plotInactivation_VInormRelation(inact_obj):
    plt.figure()
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized current')
    plt.title('Inactivation: Voltage/HMM_Normalized Current Relation')
    plt.plot(inact_obj.v_vec, inact_obj.inorm_vec, 'o', c='black')
    ssi_slope, v_half, top, bottom, tau0 = cf.calc_inact_obj(inact_obj)
    formatted_ssi_slope = np.round(ssi_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5, f'Slope: {formatted_ssi_slope}')
    plt.text(-10, 0.3, f'V50: {formatted_v_half}')
    x_values_v = np.arange(inact_obj.st_cl, inact_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, ssi_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c='red')
    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0],
                     'Plots_Folder/HMM_Inactivation Voltage Normalized Current Relation'))

def plotInactivation_VInormRelation_plt(inact_obj, plt, color):

    diff = 0
    if color == 'red':
        diff = 0.5
    plt.plot(inact_obj.v_vec, inact_obj.inorm_vec, 'o', c=color)
    ssi_slope, v_half, top, bottom = cf.calc_inact_obj(inact_obj)
    formatted_ssi_slope = np.round(ssi_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5 + diff, f'Slope: {formatted_ssi_slope}', c=color)
    plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c=color)
    x_values_v = np.arange(inact_obj.st_cl, inact_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, ssi_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c=color)
    return (formatted_v_half, formatted_ssi_slope)

def plotInactivation_TimeVRelation(inact_obj):
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title('Inactivation Time/Voltage relation')
    [plt.plot(inact_obj.t_vec, inact_obj.all_v_vec_t[i], c='black') for i in np.arange(inact_obj.L)]
    # save as PGN file
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/HMM_Inactivation Time Voltage Relation'))

def plotInactivation_TCurrDensityRelation(inact_obj):
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Inactivation Time/Current density relation')
    [plt.plot(inact_obj.t_vec[1:], inact_obj.all_is[i], c='black') for i in np.arange(inact_obj.L)]
    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Inactivation Time Current Density Relation"))

def plotInactivation_TCurrDensityRelation(inact_obj, plt, color):
    [plt.plot(inact_obj.t_vec[-800:-700], inact_obj.all_is[i][-800:-700], c=color) for i in np.arange(inact_obj.L)]

def plotInactivation_Tau_0mV(inact_obj):
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Inactivation Tau at 0 mV')
    # select 0 mV
    volt = 0  # mV
    mask = np.where(inact_obj.v_vec == volt)[0]
    curr = np.array(inact_obj.all_is)[mask][0]
    time = np.array(inact_obj.t_vec)[1:]
    # fit exp: IFit(t) = A * exp (-t/τ) + C
    ts, data, xs, ys, tau = inact_obj.find_tau0_inact(curr)
    # plot
    plt.plot(ts, data, color="black")
    plt.plot(xs, ys, color="red")
    formatted_tau = np.round(tau, decimals=3)
    plt.text(0.2, -0.01, f"Tau at 0 mV: {formatted_tau}", color='blue')
    # save as PGN file
    plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Inactivation Tau at 0 mV"))

def plotInactivation_Tau_0mV_plt(inact_obj, plt, color, upper=700):

    diff = 0
    if color == 'red':
        diff = 1.5

    def fit_expon(x, a, b, c):
        return a + b * np.exp(-1 * c * x)

    def one_phase(x, y0, plateau, k):
        return y0 + (plateau - y0) * (1 - np.exp(-k * x))

    act = Activation(channel_name=inact_obj.channel_name)
    act.clamp_at_volt(0)
    starting_index = list(act.i_vec).index(act.find_ipeaks_with_index()[1])

    t_vecc = act.t_vec[starting_index:upper]
    i_vecc = act.i_vec[starting_index:upper]
    try:
        popt, pcov = optimize.curve_fit(fit_expon, t_vecc, i_vecc, method='dogbox')
        fit = 'exp'
        tau = 1 / popt[2]
        fitted_i = fit_expon(act.t_vec[starting_index:upper], popt[0], popt[1], popt[2])
    except:
        popt, pcov = optimize.curve_fit(one_phase, t_vecc, i_vecc, method='dogbox')
        fit = 'one_phase'
        tau = 1 / popt[2]
        fitted_i = one_phase(act.t_vec[starting_index:upper], popt[0], popt[1], popt[2])

    xmid = (max(t_vecc) + min(t_vecc)) / 2
    ymid = (max(i_vecc) + min(i_vecc)) / 2
    if color == 'red':
        diff = ymid * 0.2

    plt.plot(act.t_vec[starting_index:upper], fitted_i, c=color)
    plt.plot(t_vecc, i_vecc, 'o', c=color)
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

def fit_exp(inact_obj, x, a, b, c):
    """
    IFit(t) = A * exp (-t/τ) + C
    """
    return a * np.exp(-x / b) + c

def find_tau0_inact(inact_obj, raw_data):
    # take peak curr and onwards
    min_val, mindex = min((val, idx) for (idx, val) in enumerate(raw_data[:int(0.7 * len(raw_data))]))
    padding = 15  # after peak
    data = raw_data[mindex:mindex + padding]
    ts = [0.1 * i for i in range(len(data))]  # make x values which match sample times

    # calc tau and fit exp
    popt, pcov = optimize.curve_fit(fit_exp, ts, data)  # fit exponential curve
    perr = np.sqrt(np.diag(pcov))
    # print('in ' + str(all_tau_sweeps[i]) + ' the error was ' + str(perr))
    xs = np.linspace(ts[0], ts[len(ts) - 1], 1000)  # create uniform x values to graph curve
    ys = fit_exp(xs, *popt)  # get y values
    vmax = max(ys) - min(ys)  # get diff of max and min voltage
    vt = min(ys) + .37 * vmax  # get vmax*1/e
    # tau = (np.log([(vt - popt[2]) / popt[0]]) / (-popt[1]))[0]  # find time at which curve = vt
    # Roy said tau should just be the parameter b from fit_exp
    tau = popt[1]
    return ts, data, xs, ys, tau


def plot_inact(wild_params, wild_channel_name, wild_is_HMM, mut_params, mut_channel_name, mut_is_HMM, outfile, mutant_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)
    figures = []
    
    if wild_is_HMM:
        module_name_wild = ggsdHMM
    else:
        module_name_wild = ggsd
    if mut_is_HMM:
        module_name_mut = ggsdHMM
    else:
        module_name_mut = ggsd
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized current')
    plt.title(f'Inactivation: {mutant_name}')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_inact = module_name_wild.Inactivation(channel_name=wild_channel_name)
    wt_inact.genInactivation()
    inact_v_half_wt, inact_slope_wt = wt_inact.plotInactivation_VInormRelation_plt(plt, 'black')

    set_param(mut_params, mut_is_HMM)
    mut_inact = module_name_mut.Inactivation(channel_name=mut_channel_name)
    mut_inact.genInactivation()
    inact_v_half_mut, inact_slope_mut =  mut_inact.plotInactivation_VInormRelation_plt(plt, 'red')
    
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title(f'Inactivation: {mutant_name}')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_inact = module_name_wild.Inactivation(channel_name=wild_channel_name)
    wt_inact.genInactivation()
    plotInactivation_TCurrDensityRelation(wt_inact, plt, 'black')

    set_param(mut_params, is_HMM)
    mut_inact = module_name.Inactivation(channel_name=mut_channel_name)
    mut_inact.genInactivation()
    mut_inact.plotInactivation_TCurrDensityRelation(plt, 'red')
    
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Inactivation Tau at 0 mV')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_inact = module_name.Inactivation(channel_name= wild_channel_name)
    wt_inact.genInactivation()
    wt_tau = wt_inact.plotInactivation_Tau_0mV_plt(plt, 'black')
    wt_per_cur = find_persistent_current(wild_is_HMM)

    set_param(mut_params, mut_is_HMM)
    mut_inact = module_name.Inactivation(channel_name=mut_channel_name)
    mut_inact.genInactivation()
    mut_tau = mut_inact.plotInactivation_Tau_0mV_plt(plt, 'red')
    mut_per_cur = find_persistent_current(mut_is_HMM)
    
    for fig in figures: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()
