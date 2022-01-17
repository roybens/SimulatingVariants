import generalized_genSim_shorten_time_HMM as ggsdHMM
import matplotlib.backends.backend_pdf
import eval_helper as eh
import matplotlib.pyplot as plt
import curve_fitting as cf
from scipy import optimize

def set_param(param, is_HMM):
    if is_HMM:
        eh.change_params(param, scaled=False, is_HMM=True)
    else:
        1/0
        
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

def make_act_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM, channel_name):
    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []
    
    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized conductance')
    plt.title(f'Activation: {mutant_name}')

    set_param(param_values_wt, is_HMM)
    wt_act = module_name.Activation(channel_name = channel_name)
    wt_act.genActivation()
    # (formatted_v_half, formatted_gv_slope)
    act_v_half_wt, act_slope_wt = wt_act.plotActivation_VGnorm_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_act = module_name.Activation(channel_name = channel_name)
    mut_act.genActivation()
    act_v_half_mut, act_slope_mut = mut_act.plotActivation_VGnorm_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title(f'Activation: {mutant_name} IV Curve')

    set_param(param_values_wt, is_HMM)
    wt_act = module_name.Activation(channel_name = channel_name)
    wt_act.genActivation()
    wt_act.plotActivation_IVCurve_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_act = module_name.Activation(channel_name = channel_name)
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

    set_param(param_values_wt, is_HMM)
    wt_act = module_name.Activation(channel_name = channel_name)
    wt_act.genActivation()
    wt_act.plotActivation_TCurrDensityRelation_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_act = module_name.Activation(channel_name = channel_name)
    mut_act.genActivation()
    mut_act.plotActivation_TCurrDensityRelation_plt(plt, 'red')
    
    
 ############################################################################################################

    set_param(param_values_wt, is_HMM)
    wt_peak_amp = find_peak_amp(channel_name, is_HMM)

    set_param(new_params, is_HMM)
    mut_peak_amp = find_peak_amp(channel_name, is_HMM)
    
    
############################################################################################################    
    peak_amp_dict = read_peak_amp_dict()
    
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


def make_inact_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM, channel_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []
    
    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd
    
    
    ############################################################################################################

    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized current')
    plt.title(f'Inactivation: {mutant_name}')

    set_param(param_values_wt, is_HMM)
    wt_inact = module_name.Inactivation(channel_name = channel_name)
    wt_inact.genInactivation()
    inact_v_half_wt, inact_slope_wt = wt_inact.plotInactivation_VInormRelation_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_inact = module_name.Inactivation(channel_name = channel_name)
    mut_inact.genInactivation()
    inact_v_half_mut, inact_slope_mut =  mut_inact.plotInactivation_VInormRelation_plt(plt, 'red')


    ############################################################################################################
    # figures.append(plt.figure())
    # plt.xlabel('Time $(ms)$')
    # plt.ylabel('Voltage $(mV)$')
    # plt.title('Inactivation Time/Voltage relation')

    # set_param(param_values_wt, is_HMM)
    # wt_inact = module_name.Inactivation(channel_name = channel_name)
    # wt_inact.genInactivation()
    # wt_inact.plotInactivation_TimeVRelation_plt(plt, 'black')

    # set_param(new_params, is_HMM)
    # mut_inact = module_name.Inactivation(channel_name = channel_name)
    # mut_inact.genInactivation()
    # mut_inact.plotInactivation_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title(f'Inactivation: {mutant_name}')

    set_param(param_values_wt, is_HMM)
    wt_inact = module_name.Inactivation(channel_name = channel_name)
    wt_inact.genInactivation()
    wt_inact.plotInactivation_TCurrDensityRelation(plt, 'black')

    set_param(new_params, is_HMM)
    mut_inact = module_name.Inactivation(channel_name = channel_name)
    mut_inact.genInactivation()
    mut_inact.plotInactivation_TCurrDensityRelation(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title(f'Inactivation Tau at 0 mV: {mutant_name}')

    set_param(param_values_wt, is_HMM)
    wt_inact = module_name.Inactivation(channel_name = channel_name)
    wt_inact.genInactivation()
    wt_tau = wt_inact.plotInactivation_Tau_0mV_plt(plt, 'black')
    wt_per_cur = find_persistent_current(is_HMM)

    set_param(new_params, is_HMM)
    mut_inact = module_name.Inactivation(channel_name = channel_name)
    mut_inact.genInactivation()
    mut_tau = mut_inact.plotInactivation_Tau_0mV_plt(plt, 'red')
    mut_per_cur = find_persistent_current(is_HMM)
    
    

    
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