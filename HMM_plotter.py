import generalized_genSim_shorten_time_HMM as ggsdHMM
import matplotlib.backends.backend_pdf
import eval_helper as eh
import matplotlib.pyplot as plt
# other
import curve_fitting as cf
from scipy import optimize
import argparse
import generalized_genSim_shorten_time as ggsd

def set_param(param, is_HMM,sim_obj = None):
    if is_HMM:
        eh.change_params(param, scaled=False, is_HMM=True,sim_obj = sim_obj)
    else:
        1/0
        
def read_peak_amp_dict():
    return {"T400RAdult": 0.645, "I1640NAdult": 0.24, "m1770LAdult": 0.4314, "neoWT": 0.748, "T400RAneo": 0.932, "I1640NNeo": 0.28, "m1770LNeo": 1, "K1260E" : 1, "A427D" : 1}
        
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

def make_act_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM, channel_name,channel_name_HH = None):
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
    wt_act = module_name.Activation(channel_name = channel_name)
    if channel_name_HH:
        wt_act = ggsd.Activation(channel_name = channel_name_HH)
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj = wt_act)
    wt_act.genActivation()
    print('1')
    # (formatted_v_half, formatted_gv_slope)
    act_v_half_wt, act_slope_wt = wt_act.plotActivation_VGnorm_plt(plt, 'black')
    print('2')

   
    mut_act = module_name.Activation(channel_name = channel_name)
    
    set_param(new_params, is_HMM,sim_obj = mut_act)
    mut_act.genActivation()
    print('3')
    act_v_half_mut, act_slope_mut = mut_act.plotActivation_VGnorm_plt(plt, 'red')
    print('4')
    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title(f'Activation: {mutant_name} IV Curve')

    
    wt_act = module_name.Activation(channel_name = channel_name)
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj = wt_act)
    wt_act.genActivation()
    wt_act.plotActivation_IVCurve_plt(plt, 'black')

    mut_act = module_name.Activation(channel_name = channel_name)
    set_param(new_params, is_HMM,sim_obj = mut_act)
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

    
    wt_act = module_name.Activation(channel_name = channel_name)
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj = wt_act)
    wt_act.genActivation()
    wt_act.plotActivation_TCurrDensityRelation_plt(plt, 'black')
    wt_peak_amp = find_peak_amp(channel_name, is_HMM)

    
    mut_act = module_name.Activation(channel_name = channel_name)
    set_param(new_params, is_HMM, sim_obj = mut_act)
    mut_act.genActivation()
    mut_act.plotActivation_TCurrDensityRelation_plt(plt, 'red')
    mut_peak_amp = find_peak_amp(channel_name, is_HMM)
    
    
 ############################################################################################################
  
   
    

   
############################################################################################################    
    peak_amp_dict = read_peak_amp_dict()
    
    figures.append(plt.figure())
    if mutant_protocol_csv_name is not None:
        goal_dict = read_mutant_protocols(mutant_protocol_csv_name, mutant_name)
        plt.text(0.4,0.9,"(actual, goal)")
        plt.text(0.1,0.7,"activation v half: " + str((act_v_half_mut - act_v_half_wt , goal_dict['dv_half_act'])))
        plt.text(0.1,0.5,"activation slope: " + str((act_slope_mut/act_slope_wt , goal_dict['gv_slope']/100)))
        plt.text(0.1,0.3,"peak amp: " + str((mut_peak_amp/wt_peak_amp , peak_amp_dict[mutant_name])))
        print("(actual, goal)")
        print("activation v half: " + str((act_v_half_mut - act_v_half_wt , goal_dict['dv_half_act'])))
        print("activation slope: " + str((act_slope_mut/act_slope_wt , goal_dict['gv_slope']/100)))

    

    plt.axis('off')
    for fig in figures: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()

    ############################################################################################################
    
    


def make_inact_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM, channel_name,channel_name_HH = None):
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
    wt_inact = module_name.Inactivation(channel_name = channel_name)
    if channel_name_HH:
        wt_inact = ggsd.Inactivation(channel_name = channel_name_HH)
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj = wt_inact)
    wt_inact.genInactivation()
    inact_v_half_wt, inact_slope_wt = wt_inact.plotInactivation_VInormRelation_plt(plt, 'black')

    mut_inact = module_name.Inactivation(channel_name = channel_name)
    set_param(new_params, is_HMM, sim_obj = mut_inact)
    mut_inact.genInactivation()
    inact_v_half_mut, inact_slope_mut = mut_inact.plotInactivation_VInormRelation_plt(plt, 'red')


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

    wt_inact = module_name.Inactivation(channel_name = channel_name)
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj = wt_inact)
    wt_inact.genInactivation()
    wt_inact.plotInactivation_TCurrDensityRelation(plt, 'black')

    mut_inact = module_name.Inactivation(channel_name = channel_name)
    set_param(new_params, is_HMM, sim_obj = mut_inact)
    mut_inact.genInactivation()
    mut_inact.plotInactivation_TCurrDensityRelation(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title(f'Inactivation Tau at 0 mV: {mutant_name}')

    wt_inact = module_name.Inactivation(channel_name = channel_name)
    wt_inact.genInactivation()
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj = wt_inact)
    wt_tau = wt_inact.plotInactivation_Tau_0mV_plt(plt, 'black')
    wt_per_cur = find_persistent_current(is_HMM)

    mut_inact = module_name.Inactivation(channel_name = channel_name)
    set_param(new_params, is_HMM, sim_obj = mut_inact)
    mut_inact.genInactivation()
    mut_tau = mut_inact.plotInactivation_Tau_0mV_plt(plt, 'red')
    mut_per_cur = find_persistent_current(is_HMM)
    
    

    
    figures.append(plt.figure())
    if mutant_protocol_csv_name is not None:
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

def make_recov_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM, channel_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []
    fig = plt.figure(figsize=(5, 20))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    #ax5 = fig.add_subplot(6, 1, 5)
    #ax6 = fig.add_subplot(6, 1, 6)
    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd
    figures.append(plt.figure())
    
    wt_recov = module_name.RFI(channel_name=channel_name)
    if param_values_wt is not None:
        set_param(param_values_wt, is_HMM,sim_obj =wt_recov )
    wt_recov.genRecInactTau()
    wt_recov.plotAllRFI(ax1, ax2, ax3, ax4, 'black')
    
    #set_param(new_params, is_HMM)
    
    #mut_recov = module_name.RFI(channel_name=channel_name)
    
    #set_param(new_params, is_HMM, sim_obj = mut_recov)
    #mut_recov.genRecInactTau() 
    #mut_recov.clampRecInactTau(5000)
    #mut_recov.plotAllRFI(ax1, ax2, ax5, ax6, 'red')
    

def make_ramp_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM, channel_name):
    """
    input:
        new_params: a set of variant parameters
        param_values_wt: WT parameters. Defaulted to NA 16 WT.
        filename: name of the pdf file into which we want to store the figures
    return:
        none; creates plots for ramp
    """
    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd

    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)


    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []

    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title(f'Ramp: {mutant_name}')

    set_param(param_values_wt, is_HMM)
    wt_ramp = module_name.Ramp(channel_name = channel_name)
    wt_ramp.genRamp()
    wt_ramp.plotRamp_TimeVRelation_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_ramp = module_name.Ramp(channel_name = channel_name)
    mut_ramp.genRamp()
    mut_ramp.plotRamp_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    f.add_subplot(111, frameon=False)  # for shared axes labels and big title
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current', labelpad=25)
    plt.title(f"Ramp: {mutant_name} Time Current Density Relation", x=0.4, y=1.1)
    ax1.set_title("Ramp AUC")
    ax2.set_title("Persistent Current")

    set_param(param_values_wt, is_HMM)
    wt_ramp = module_name.Ramp(channel_name = channel_name)
    wt_ramp.genRamp()
    wt_ramp_area, wt_ramp_persistcurr = wt_ramp.plotRamp_TimeCurrentRelation_plt(ax1, ax2, 'black')

    set_param(new_params, is_HMM)
    mut_ramp = module_name.Ramp(channel_name = channel_name)
    mut_ramp.genRamp()
    mut_ramp_area, mut_ramp_persistcurr =mut_ramp.plotRamp_TimeCurrentRelation_plt(ax1, ax2, 'red')

    plt.tight_layout()

    figures.append(f)
    ############################################################################################################
    figures.append(plt.figure())
    goal_dict = read_mutant_protocols(mutant_protocol_csv_name, mutant_name)
    plt.text(0.4,0.9,"(actual, goal)")
    plt.text(0.1,0.7,"area under curve: " + str((mut_ramp_area/wt_ramp_area , goal_dict['ramp']/100)))
    plt.text(0.1,0.5,"persistent current: " + str((mut_ramp_persistcurr/wt_ramp_persistcurr, goal_dict['persistent']/100)))

    plt.axis('off')
    for fig in figures:
        pdf.savefig( fig )
    pdf.close()


############################################################################################################
def make_UDB20_plots(new_params, mutant_name, mutant_protocol_csv_name, param_values_wt, filename, is_HMM,
                     channel_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    figures = []

    if is_HMM:
        module_name = ggsdHMM
    else:
        module_name = ggsd

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title(f"UBD20: Time Voltage Relation for {mutant_name}")

    set_param(param_values_wt, is_HMM)
    wt_udb20 = module_name.UDB20(channel_name=channel_name)
    wt_udb20.genUDB20()
    wt_udb20.plotUDB20_TimeVRelation_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_udb20 = module_name.UDB20(channel_name=channel_name)
    mut_udb20.genUDB20()
    mut_udb20.plotUDB20_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    figures.append(plt.figure())
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current $(pA)$')
    plt.title(f"UDB20: Current of Pulses for {mutant_name}")

    set_param(param_values_wt, is_HMM)
    wt_udb20 = module_name.UDB20(channel_name=channel_name)
    wt_udb20.genUDB20()
    wt_peakCurrs5 = wt_udb20.getPeakCurrs()
    wt_udb20.plotUDB20_TimeCurrentRelation_plt(plt, 'black')

    set_param(new_params, is_HMM)
    mut_udb20 = module_name.UDB20(channel_name=channel_name)
    mut_udb20.genUDB20()
    mut_peakCurrs5 = mut_udb20.getPeakCurrs()
    mut_udb20.plotUDB20_TimeCurrentRelation_plt(plt, 'red')

    ############################################################################################################

    figures.append(plt.figure())
    goal_dict = read_mutant_protocols(mutant_protocol_csv_name, mutant_name)
    plt.text(0.4, 0.9, "(actual, goal)")
    plt.text(0.1, 0.7, "peak5/peak1: " + str((mut_peakCurrs5 / wt_peakCurrs5, goal_dict['udb20'] / 100)))
    plt.axis('off')
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()




#######################
# MAIN
#######################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulated data.')
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    p = [1.6145008130686316,
         1.2702355752969856,
         0.2856140201135051,
         2.000672353749617,
         159.19293105141264,
         0.8882089670901088,
         1.54307338742142,
         4.835533385345919,
         184.46766214071704,
         0.6193119174876813,
         8.851518497666747,
         0.07019281223744751,
         46.30970872218895,
         12.027049656918223,
         1.0303204433640094,
         0.05027526734333132,
         1791.9670172949814,
         1.3053734595552096,
         20.37380422148677,
         -9.174778056184731]

    eh.change_params(p, scaled=False, is_HMM=True)

    if args.function == 1:
        make_UDB20_plots(p, "K1260E", "./csv_files/mutant_protocols.csv", p, "./Plots_Folder/jinan_test.pdf", is_HMM=True,
                       channel_name="na12mut")
