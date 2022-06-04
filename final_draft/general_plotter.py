import os
import matplotlib.backends.backend_pdf
import eval_helper as eh
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltt
import curve_fitting as cf
import numpy as np
from scipy import optimize
import generate_simulation

module_name = generate_simulation


def set_param(param, is_HMM, sim_obj):
    eh.change_params(param, scaled=False, is_HMM=is_HMM, sim_obj=sim_obj)


def read_peak_amp_dict():
    return {"T400RAdult": 0.645, "I1640NAdult": 0.24, "m1770LAdult": 0.4314, "neoWT": 0.748, "T400RAneo": 0.932,
            "I1640NNeo": 0.28, "m1770LNeo": 1, "K1260E": 1}


def read_mutant_protocols(mutant_protocols_csv, mutant):
    '''
    Reads data for a single MUTANT from a csv of mutant protocols.
    Returns a dictionary with all the relevant protocols for that 
    MUTANT.
    '''
    lines = []
    with open(mutant_protocols_csv, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]

    # Each line[0] except the first should contain the name of the mutant
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

##From old code
def find_persistent_current(is_HMM):
    """
    returns the persistent current, gieven that the NEURON model already has parameters properly set
    """

    ramp = module_name.Ramp()
    ramp.genRamp()
    return ramp.persistentCurrent()
##From old code

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


def plotActivation_IVCurve_plt(act_obj, plt_in, color):
    plt_in.plot(np.array(act_obj.v_vec), np.array(act_obj.ipeak_vec), 'o', c=color)
    # plt.text(-110, -0.05, 'Vrev at ' + str(round(act_obj.vrev, 1)) + ' mV', fontsize=10, c='blue')
    formatted_peak_i = np.round(min(act_obj.ipeak_vec), decimals=2)
    # plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps


def plotActivation_TimeVRelation(act_obj):
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title('Activation Time/Voltage relation')
    [plt.plot(act_obj.t_vec, act_obj.all_v_vec_t[i], c='black') for i in np.arange(act_obj.L)]
    # save as PGN file
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/HMM_Activation Time Voltage Relation'))


def plotActivation_TCurrDensityRelation(act_obj, plt_in=None, color='black', xlim=None):
    if plt_in is None:
        pltt.figure()
    else:
        plt = plt_in
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Activation Time/Current density relation')
    curr = np.array(act_obj.all_is)
    [plt.plot(act_obj.t_vec[1:], curr[i], c=color) for i in np.arange(len(curr))]
    if xlim is not None:
        for i in np.arange(len(curr)):
            plt.plot(act_obj.t_vec[1:], curr[i], c=color)
            plt.xlim(xlim)

    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Activation Time Current Density Relation"))


# def plotActivation_TCurrDensityRelation_plt(act_obj,plt,color):
#    curr = np.array(act_obj.all_is)
#    mask = np.where(np.logical_or(act_obj.v_vec == 0, act_obj.v_vec == 10))
#    [plt.plot(act_obj.t_vec[190:300], curr[i][190:300], c=color) for i in np.arange(len(curr))[mask]]


def plotActivation_VGnorm_plt(act_obj, plt, color):
    """
    Saves activation plot as PGN file.
    """

    diff = 0
    if color == 'red':
        diff = 0.5

    plt.plot(act_obj.v_vec, act_obj.gnorm_vec, 'o', c=color)
    gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj)
    # gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj.channel_name, True)
    formatted_gv_slope = np.round(gv_slope, decimals=2)
    formatted_v_half = np.round(v_half, decimals=2)
    plt.text(-10, 0.5 + diff, f'Slope: {formatted_gv_slope}', c=color)
    plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c=color)
    x_values_v = np.arange(act_obj.st_cl, act_obj.end_cl, 1)
    curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
    plt.plot(x_values_v, curve, c=color)
    return (formatted_v_half, formatted_gv_slope)


def plotActivation_Tau_0mV_plt(act_obj, plt, color, upper=700):
    diff = 0
    if color == 'red':
        diff = 1.5

    def fit_expon(x, a, b, c):
        return a + b * np.exp(-1 * c * x)

    def one_phase(x, y0, plateau, k):
        return y0 + (plateau - y0) * (1 - np.exp(-k * x))

    act_obj.clamp(0)
    starting_index = list(act_obj.i_vec).index(act_obj.find_ipeaks_with_index()[1])

    t_vecc = act_obj.t_vec[starting_index:upper]
    i_vecc = act_obj.i_vec[starting_index:upper]
    try:
        popt, pcov = optimize.curve_fit(fit_expon, t_vecc, i_vecc, method='dogbox')
        fit = 'exp'
        tau = 1 / popt[2]
        fitted_i = fit_expon(act_obj.t_vec[starting_index:upper], popt[0], popt[1], popt[2])
    except:
        popt, pcov = optimize.curve_fit(one_phase, t_vecc, i_vecc, method='dogbox')
        fit = 'one_phase'
        tau = 1 / popt[2]
        fitted_i = one_phase(act_obj.t_vec[starting_index:upper], popt[0], popt[1], popt[2])

    xmid = (max(t_vecc) + min(t_vecc)) / 2
    ymid = (max(i_vecc) + min(i_vecc)) / 2
    if color == 'red':
        diff = ymid * 0.2

    plt.plot(act_obj.t_vec[starting_index:upper], fitted_i, c=color)
    plt.plot(t_vecc, i_vecc, 'o', c=color)
    plt.text(xmid, ymid + diff, f"Tau at 0 mV: {tau}", color=color)

    return tau


def plot_prst_curr(act_obj, plt_in=None, v_trace=0, color='black'):
    if plt_in is None:
        plt = pltt
    else:
        plt = plt_in
    diff = 0
    prst_cur, t_mask = act_obj.get_perst_curr(v_trace)
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title(f'last 10% of {v_trace}mV Step ')
    x_time = act_obj.t_vec[t_mask]
    y_curr = prst_cur
    xmid = (max(x_time) + min(x_time)) / 2
    ymid = (max(y_curr) + min(y_curr)) / 2
    if color == 'red':
        diff = ymid * 0.2
    plt.plot(x_time,y_curr , c=color)
    mean_prst = cf.calc_act_prst_curr(act_obj,v_trace=v_trace)
    plt.text(xmid, ymid + diff, f"prst curr {v_trace} mV: {mean_prst}", color=color)


def plot_currents_general(act_obj, plt_in=None, time_range=None, v_range=None, i_range=None, color='black'):
    if plt_in is None:
        plt = pltt
    else:
        plt = plt_in
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Activation Time/Current density relation')
    if v_range:
        v_mask = list(np.where(np.logical_and(act_obj.v_vec >= v_range[0], act_obj.v_vec <= v_range[1]))[0])
    else:
        v_mask = list(range(len(act_obj.v_vec)))
    if time_range:
        t_mask = act_obj.get_time_inds(time_range[0], time_range[1])
    else:
        t_mask = range(len(act_obj.t_vec))
    [plt.plot(act_obj.t_vec[t_mask], act_obj.all_is[i][t_mask], c=color) for i in v_mask]
    if i_range:
        plt.ylim(i_range)
    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Activation Time Current Density Relation"))


def plot_act(wild_params, wild_channel_name, wild_is_HMM, mut_params, mut_channel_name, mut_is_HMM, outfile,
             mutant_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)
    figures = []
    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized conductance')
    plt.title(f'Activation: {mutant_name}')
    wt_act = module_name.Activation_general(channel_name=wild_channel_name)
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM, sim_obj=wt_act)

    wt_act.genActivation()
    # (formatted_v_half, formatted_gv_slope)
    act_v_half_wt, act_slope_wt = plotActivation_VGnorm_plt(wt_act, plt, 'black')

    mut_act = module_name.Activation_general(channel_name=mut_channel_name)
    set_param(mut_params, mut_is_HMM, sim_obj=mut_act)
    mut_act.genActivation()
    act_v_half_mut, act_slope_mut = plotActivation_VGnorm_plt(mut_act, plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title(f'Activation: {mutant_name} IV Curve')
    plotActivation_IVCurve_plt(wt_act, plt, 'black')
    plotActivation_IVCurve_plt(mut_act, plt, 'red')

    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('I $(mA/cm^2)$')
    plt.title(f'Activation waveform at 0mV: {mutant_name}')
    plotActivation_TCurrDensityRelation(wt_act, plt_in=plt, color='black', xlim=[4.9, 10])
    plotActivation_TCurrDensityRelation(mut_act, plt_in=plt, color='red', xlim=[4.9, 10])

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    wt_tau = plotActivation_Tau_0mV_plt(wt_act, plt, 'black')
    # wt_per_cur = find_persistent_current(wild_is_HMM)
    mut_tau = plotActivation_Tau_0mV_plt(mut_act, plt, 'red')
    ############################################################################################################
    for fig in figures:  ## will open an empty extra figure :(
        pdf.savefig(fig)
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


def plotInactivation_TCurrDensityRelation(inact_obj, plt_in=None, color='black', padding=3):
    if plt_in is None:
        plt.figure()
    else:
        plt = plt_in
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Inactivation Time/Current density relation')
    inds_arr = [i for i in range(len(inact_obj.t_vec)) if
                (inact_obj.t_vec[i] > (inact_obj.st_step_time - padding / 2)) & (
                            inact_obj.t_vec[i] < (inact_obj.st_step_time + padding * 2))]
    inds_arr = np.array(inds_arr[:-1])
    time_arr = np.array(inact_obj.t_vec)[inds_arr.astype(int)]

    [plt.plot(time_arr, np.array(inact_obj.all_is[i])[inds_arr.astype(int)], c=color) for i in np.arange(inact_obj.L)]
    # save as PGN file
    plt.savefig(
        os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Inactivation Time Current Density Relation"))


# def plotInactivation_TCurrDensityRelation(inact_obj, plt, color):
#    [plt.plot(inact_obj.t_vec[-800:-700], inact_obj.all_is[i][-800:-700], c=color) for i in np.arange(inact_obj.L)]


def plot_inact(wild_params, wild_channel_name, wild_is_HMM, mut_params, mut_channel_name, mut_is_HMM, outfile,
               mutant_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)
    figures = []

    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized current')
    plt.title(f'Inactivation: {mutant_name}')
    if wild_params is not None:
        set_param(wild_params, wild_is_HMM)
    wt_inact = module_name.Inactivation_general(channel_name=wild_channel_name)
    wt_inact.genInactivation()
    inact_v_half_wt, inact_slope_wt = plotInactivation_VInormRelation_plt(wt_inact, plt, 'black')

    mut_inact = module_name.Inactivation_general(channel_name=mut_channel_name)
    set_param(mut_params, mut_is_HMM, mut_inact)
    mut_inact.genInactivation()
    inact_v_half_mut, inact_slope_mut = plotInactivation_VInormRelation_plt(mut_inact, plt, 'red')

    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title(f'Inactivation: {mutant_name}')
    plotInactivation_TCurrDensityRelation(wt_inact, plt, 'black')
    plotInactivation_TCurrDensityRelation(mut_inact, plt, 'red')

    for fig in figures:  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()
