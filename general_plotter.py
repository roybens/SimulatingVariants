import generalized_genSim_shorten_time_HMM as ggsdHMM
import generalized_genSim_shorten_time as ggsd
import matplotlib.backends.backend_pdf
import eval_helper as eh
import matplotlib.pyplot as plt
import curve_fitting as cf
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
    act_v_half_wt, act_slope_wt = wt_act.plotActivation_VGnorm_plt(plt, 'black')

    set_param(mut_params, mut_is_HMM)
    mut_act = module_name_mut.Activation(channel_name=mut_channel_name)
    mut_act.genActivation()
    act_v_half_mut, act_slope_mut = mut_act.plotActivation_VGnorm_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title(f'Activation: {mutant_name} IV Curve')

    set_param(wild_params, wild_is_HMM)
    wt_act = module_name_wild.Activation(channel_name=wild_channel_name)
    wt_act.genActivation()
    wt_act.plotActivation_IVCurve_plt(plt, 'black')

    set_param(mut_params, mut_is_HMM)
    mut_act = module_name_mut.Activation(channel_name=mut_channel_name)
    mut_act.genActivation()
    mut_act.plotActivation_IVCurve_plt(plt, 'red')

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

    set_param(wild_params, wild_is_HMM)
    wt_act = module_name_wild.Activation(channel_name=wild_channel_name)
    wt_act.genActivation()
    wt_act.plotActivation_TCurrDensityRelation_plt(plt, 'black')

    set_param(mut_params, is_HMM)
    mut_act = module_name_mut.Activation(channel_name=mut_channel_name)
    mut_act.genActivation()
    mut_act.plotActivation_TCurrDensityRelation_plt(plt, 'red')
    
    
 ############################################################################################################

    set_param(wild_params, wild_is_HMM)
    wt_peak_amp = find_peak_amp(wild_channel_name, wild_is_HMM)

    set_param(mut_params, mut_is_HMM)
    mut_peak_amp = find_peak_amp(mut_channel_name, mut_is_HMM)
    
    
############################################################################################################    
    for fig in figures: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()

    ############################################################################################################
    
    
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
    
    set_param(wild_params, wild_is_HMM)
    wt_inact = module_name_wild.Inactivation(channel_name=wild_channel_name)
    wt_inact.genInactivation()
    wt_inact.plotInactivation_TCurrDensityRelation(plt, 'black')

    set_param(mut_params, is_HMM)
    mut_inact = module_name.Inactivation(channel_name=mut_channel_name)
    mut_inact.genInactivation()
    mut_inact.plotInactivation_TCurrDensityRelation(plt, 'red')
    
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Inactivation Tau at 0 mV')
    
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
