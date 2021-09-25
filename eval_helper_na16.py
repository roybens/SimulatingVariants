########################################################
#### Important helper functions for the evaluators. ####
#### Authors: Michael Lam, Jinan Jiang #################
########################################################
import generalized_genSim_shorten_time as ggsd
import generalized_genSim_shorten_time_HMM as ggsdHMM
import matplotlib.pyplot as plt
import curve_fitting as cf

currh = ggsd.Activation(channel_name = 'na16').h
def set_param(param_values):
    

    currh.sh_na16 = param_values[0]
    currh.tha_na16 = param_values[1]
    currh.qa_na16 = param_values[2]
    currh.Ra_na16 = param_values[3]
    currh.Rb_na16 = param_values[4]
    currh.thi1_na16 = param_values[5]
    currh.thi2_na16 = param_values[6]
    currh.qd_na16 = param_values[7]
    currh.qg_na16 = param_values[8]
    currh.mmin_na16 = param_values[9]
    currh.hmin_na16 = param_values[10]
    currh.q10_na16 = param_values[11]
    currh.Rg_na16 = param_values[12]
    currh.Rd_na16 = param_values[13]
    #currh.qq_na16 = param_values[14]
    #currh.tq_na16 = param_values[15]
    currh.thinf_na16 = param_values[14]
    currh.qinf_na16 = param_values[15]
    currh.vhalfs_na16 = param_values[16]
    currh.a0s_na16 = param_values[17]
    currh.zetas_na16 = param_values[18]
    currh.gms_na16 = param_values[19]
    currh.smax_na16 = param_values[20]
    currh.vvh_na16 = param_values[21]
    currh.vvs_na16 = param_values[22]
    currh.Ena_na16 = param_values[23]
        
    '''
    currh.sh_na16 = param_values[0]
    currh.tha_na16 = param_values[1]
    currh.qa_na16 = param_values[2]
    currh.Ra_na16 = param_values[3]
    currh.Rb_na16 = param_values[4]
    currh.thi1_na16 = param_values[5]
    currh.thi2_na16 = param_values[6]
    currh.qd_na16 = param_values[7]
    currh.qg_na16 = param_values[8]
    currh.mmin_na16 = param_values[9]
    currh.hmin_na16 = param_values[10]
    currh.q10_na16 = param_values[11]
    currh.Rg_na16 = param_values[12]
    currh.Rd_na16 = param_values[13]
    currh.qq_na16 = param_values[14]
    currh.tq_na16 = param_values[15]
    currh.thinf_na16 = param_values[16]
    currh.qinf_na16 = param_values[17]
    currh.vhalfs_na16 = param_values[18]
    currh.a0s_na16 = param_values[19]
    currh.zetas_na16 = param_values[20]
    currh.gms_na16 = param_values[21]
    currh.smax_na16 = param_values[22]
    currh.vvh_na16 = param_values[23]
    currh.vvs_na16 = param_values[24]
    currh.Ena_na16 = param_values[25]
    '''

def get_wt_params():
    # WT params

    param_values_wt = [8.0,
     -35.0,
     7.2,
     0.4,
     0.124,
     -45.0,
     -45.0,
     0.5,
     1.5,
     0.02,
     0.01,
     2.0,
     0.01,
     0.03,
     #10.0,
     #-55.0,
     -55.0,
     7.0,
     -60.0,
     0.0003,
     12.0,
     0.2,
     10.0,
     -58.0,
     2.0,
     55.0]
    
    assert len(param_values_wt) == 24, 'This does not make sense'
    
    return param_values_wt

scale_voltage = 30
scale_fact = 7.5
wt_params = get_wt_params()


def get_currh_params_str():
    param_list_str = ['currh.sh_na16', 'currh.tha_na16', 'currh.qa_na16', 'currh.Ra_na16', 'currh.Rb_na16', 'currh.thi1_na16', 'currh.thi2_na16', 'currh.qd_na16', 'currh.qg_na16', 'currh.mmin_na16', 'currh.hmin_na16', 'currh.q10_na16', 'currh.Rg_na16', 'currh.Rd_na16', 'currh.qq_na16', 'currh.tq_na16', 'currh.thinf_na16', 'currh.qinf_na16', 'currh.vhalfs_na16', 'currh.a0s_na16', 'currh.zetas_na16', 'currh.gms_na16', 'currh.smax_na16', 'currh.vvh_na16', 'currh.vvs_na16', 'currh.Ena_na16']
    return param_list_str

def get_name_params_str():
    result = []
    for param_currh in get_currh_params_str():
        result.append(param_currh[6:-5])
    return result

def get_norm_vals(new_params):
    set_param(new_params)
    
    act = ggsd.Activation(channel_name = 'na16')
    act.genActivation()
    norm_act_y_val = sorted(list(act.gnorm_vec))
    print("The normalized activation G-V data is:")
    print(norm_act_y_val)

    inact = ggsd.Inactivation(channel_name = 'na16')
    inact.genInactivation()
    norm_inact_y_val = sorted(list(inact.inorm_vec))
    print("The normalized inactivation I-V data is:")
    print(norm_act_y_val)

def get_fitted_act_conductance_arr(x_array, gv_slope, act_v_half, act_top, act_bottom):
    cond_arr = []
    for x in x_array:
        cond_arr.append(cf.boltzmann(x, gv_slope, act_v_half, act_top, act_bottom))
    return cond_arr

def get_fitted_inact_current_arr(x_array, ssi_slope, inact_v_half, inact_top, inact_bottom):
    curr_arr = []
    for x in x_array:
        curr_arr.append(cf.boltzmann(x, ssi_slope, inact_v_half, inact_top, inact_bottom))
    return curr_arr

def make_act_plots(new_params, param_values_wt = wt_params):
    
    ############################################################################################################
    plt.figure()
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized conductance')
    plt.title('Activation: Voltage/Normalized conductance')

    set_param(param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na16')
    wt_act.genActivation()
    # (formatted_v_half, formatted_gv_slope)
    act_v_half_wt, act_slope_wt = wt_act.plotActivation_VGnorm_plt(plt, 'black')

    set_param(new_params)
    mut_act = ggsd.Activation(channel_name = 'na16')
    mut_act.genActivation()
    act_v_half_mut, act_slope_mut = mut_act.plotActivation_VGnorm_plt(plt, 'red')

    ############################################################################################################
    plt.figure()
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Peak Current $(pA)$')
    plt.title("Activation: IV Curve")

    set_param(param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na16')
    wt_act.genActivation()
    wt_act.plotActivation_IVCurve_plt(plt, 'black')

    set_param(new_params)
    mut_act = ggsd.Activation(channel_name = 'na16')
    mut_act.genActivation()
    mut_act.plotActivation_IVCurve_plt(plt, 'red')

    ############################################################################################################
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title('Activation Time/Voltage relation')

    set_param(param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na16')
    wt_act.genActivation()
    wt_act.plotActivation_TimeVRelation_plt(plt, 'black')

    set_param(new_params)
    mut_act = ggsd.Activation(channel_name = 'na16')
    mut_act.genActivation()
    mut_act.plotActivation_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Activation Time/Current density relation')

    set_param(param_values_wt)
    wt_act = ggsd.Activation(channel_name = 'na16')
    wt_act.genActivation()
    wt_act.plotActivation_TCurrDensityRelation_plt(plt, 'black')

    set_param(new_params)
    mut_act = ggsd.Activation(channel_name = 'na16')
    mut_act.genActivation()
    mut_act.plotActivation_TCurrDensityRelation_plt(plt, 'red')

    ############################################################################################################
    goal_dict = read_mutant_protocols('mutant_protocols.csv', 'NA16_MUT')
    print("(actual, goal)")
    print("activation v half: " + str((act_v_half_mut - act_v_half_wt , goal_dict['dv_half_act'])))
    print("activation slope: " + str((act_slope_mut/act_slope_wt , goal_dict['gv_slope']/100)))


def make_inact_plots(new_params, param_values_wt = wt_params):
    
    ############################################################################################################

    plt.figure()
    plt.xlabel('Voltage $(mV)$')
    plt.ylabel('Normalized current')
    plt.title('Inactivation: Voltage/Normalized Current Relation')

    set_param(param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na16')
    wt_inact.genInactivation()
    inact_v_half_wt, inact_slope_wt = wt_inact.plotInactivation_VInormRelation_plt(plt, 'black')

    set_param(new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na16')
    mut_inact.genInactivation()
    inact_v_half_mut, inact_slope_mut =  mut_inact.plotInactivation_VInormRelation_plt(plt, 'red')


    ############################################################################################################
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title('Inactivation Time/Voltage relation')

    set_param(param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na16')
    wt_inact.genInactivation()
    wt_inact.plotInactivation_TimeVRelation_plt(plt, 'black')

    set_param(new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na16')
    mut_inact.genInactivation()
    mut_inact.plotInactivation_TimeVRelation_plt(plt, 'red')

    ############################################################################################################
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.title('Inactivation Time/Voltage relation')

    set_param(param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na16')
    wt_inact.genInactivation()
    wt_inact.plotInactivation_TCurrDensityRelation(plt, 'black')

    set_param(new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na16')
    mut_inact.genInactivation()
    mut_inact.plotInactivation_TCurrDensityRelation(plt, 'red')

    ############################################################################################################
    plt.figure()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Current density $(mA/cm^2)$')
    plt.title('Inactivation Tau at 0 mV')

    set_param(param_values_wt)
    wt_inact = ggsd.Inactivation(channel_name = 'na16')
    wt_inact.genInactivation()
    wt_inact.plotInactivation_Tau_0mV_plt(plt, 'black')

    set_param(new_params)
    mut_inact = ggsd.Inactivation(channel_name = 'na16')
    mut_inact.genInactivation()
    mut_inact.plotInactivation_Tau_0mV_plt(plt, 'red')

    goal_dict = read_mutant_protocols('mutant_protocols.csv', 'NA16_MUT')
    print("(actual, goal)")
    print("inactivation v half: " + str((inact_v_half_mut - inact_v_half_wt , goal_dict['dv_half_ssi'])))
    print("inactivation slope: " + str((inact_slope_mut/inact_slope_wt , goal_dict['ssi_slope']/100)))


def get_param_list_in_str():
    l1 = "[currh.sh_na16, currh.tha_na16, currh.qa_na16, currh.Ra_na16, currh.Rb_na16, currh.thi1_na16, currh.thi2_na16,"
    l2 = " currh.qd_na16, currh.qg_na16, currh.mmin_na16, currh.hmin_na16, currh.q10_na16, currh.Rg_na16, currh.Rd_na16,"
    l3 = " currh.qq_na16, currh.tq_na16, currh.thinf_na16, currh.qinf_na16, currh.vhalfs_na16, currh.a0s_na16,"
    l4 = " currh.zetas_na16, currh.gms_na16, currh.smax_na16, currh.vvh_na16, currh.vvs_na16, currh.Ena_na16]"
    param_list = l1 + l2 + l3 + l4
    return param_list

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


def read_params_range(param_range_csv):
    
    with open(param_range_csv, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]
    
    result = {}
    for line in lines[1:]:
        result[line[0]] = (float(line[1]), float(line[2]), float(line[3][:-1]))
    
    return result
    
    


def change_params(new_params, scaled=False, is_HMM=False):
    '''
    Change params on Na12mut channel in NEURON.
    ---
    Param new_params_scaled: list of param values
        scaled: whether the parameters are scaled to be between 0 and 1
    '''

    change_params_dict(new_params)

    return


def scale_params_dict(down, params_arr):
    '''
    Scale parameters between 0 and 1.
    ---
    Param down: boolean to determine whether to scale down or up
    Param params: list of param values to scale
    Return: list of scaled param values
    '''
    
    #original values of the paramter
    base_value = {
    'Ena_na12mut': 55,
    'Rd_na12mut': .03,
    'Rg_na12mut': .01,
    'Rb_na12mut': .124,
    'Ra_na12mut': 0.4,
    'a0s_na12mut': 0.0003,
    'gms_na12mut': .02,
    'hmin_na12mut': .01,
    'mmin_na12mut': .02,
    'qinf_na12mut': 7,
    'q10_na12mut': 2,
    'qg_na12mut': 1.5,
    'qd_na12mut': .5,
    'qa_na12mut': 7.2,
    'smax_na12mut': 10,
    'sh_na12mut': 8,
    'thinf_na12mut': -45,
    'thi2_na12mut': -45,
    'thi1_na12mut': -45,
    'tha_na12mut': -30,
    'vvs_na12mut': 2,
    'vvh_na12mut': -58,
    'vhalfs_na12mut': -60,
    'zetas_na12mut': 12
    }
    types = {
    'Ena_na12mut': 'p',
    'Rd_na12mut': 'p',
    'Rg_na12mut': 'p',
    'Rb_na12mut': 'p',
    'Ra_na12mut': 'p',
    'a0s_na12mut': 'md',
    'gms_na12mut': 'p',
    'hmin_na12mut': 'p',
    'mmin_na12mut': 'p',
    'qinf_na12mut': 'md',
    'q10_na12mut': 'p',
    'qg_na12mut': 'md',
    'qd_na12mut': 'md',
    'qa_na12mut': 'md',
    'smax_na12mut': 'p',
    'sh_na12mut': 'p',
    'thinf_na12mut': 'p',
    'thi2_na12mut': 'p',
    'thi1_na12mut': 'p',
    'tha_na12mut': 'p',
    'vvs_na12mut': 'p',
    'vvh_na12mut': 'p',
    'vhalfs_na12mut': 'p',
    'zetas_na12mut': 'p'
    }
    inds = {
    'Ena_na12mut': 0,
    'Rd_na12mut': 1,
    'Rg_na12mut': 2,
    'Rb_na12mut': 3,
    'Ra_na12mut': 4,
    'a0s_na12mut': 5,
    'gms_na12mut': 6,
    'hmin_na12mut': 7,
    'mmin_na12mut': 8,
    'qinf_na12mut': 9,
    'q10_na12mut': 10,
    'qg_na12mut': 11,
    'qd_na12mut': 12,
    'qa_na12mut': 13,
    'smax_na12mut': 14,
    'sh_na12mut': 15,
    'thinf_na12mut': 16,
    'thi2_na12mut': 17,
    'thi1_na12mut': 18,
    'tha_na12mut': 19,
    'vvs_na12mut': 20,
    'vvh_na12mut': 21,
    'vhalfs_na12mut': 22,
    'zetas_na12mut': 23
    }


    params_dict = {}
    bounds = {}
    for k, v in base_value.items():
        #print(f'k is {k} inds[k] is {inds[k]}')
        params_dict[k] = params_arr[inds[k]]
        val_type = types[k]
        if val_type == 'md': #scale kinetic param
            bounds[k] = (v/scale_fact, v*scale_fact)
        elif val_type == 'p': #scale voltage param
            bounds[k] = (v-scale_voltage, v+scale_voltage)
        else:
            bounds[k]= (0,1)
    
    if down:
        return [(v-bounds[k][0])/(bounds[k][1]-bounds[k][0]) for k,v in params_dict.items()]

    new_params = {}
    for  k,v  in params_dict.items():
        new_params[k]= v*(bounds[k][1]-bounds[k][0]) + bounds[k][0]
    #print(new_params)
    return new_params

def change_params_dict(new_params):
    '''
    Change params on Na12mut channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    
    #get NEURON h
    currh = ggsd.Activation().h
    #change values of params
    currh.Rd_na12mut= new_params['Rd_na12mut']
    currh.Rg_na12mut= new_params['Rg_na12mut']
    currh.Rb_na12mut= new_params['Rb_na12mut']
    currh.Ra_na12mut= new_params['Ra_na12mut']
    currh.a0s_na12mut= new_params['a0s_na12mut']
    currh.gms_na12mut= new_params['gms_na12mut']
    currh.hmin_na12mut= new_params['hmin_na12mut']
    currh.mmin_na12mut= new_params['mmin_na12mut']
    currh.qinf_na12mut= new_params['qinf_na12mut']
    currh.q10_na12mut= new_params['q10_na12mut']
    currh.qg_na12mut= new_params['qg_na12mut']
    currh.qd_na12mut= new_params['qd_na12mut']
    currh.qa_na12mut= new_params['qa_na12mut']
    currh.smax_na12mut= new_params['smax_na12mut']
    currh.sh_na12mut= new_params['sh_na12mut']
    currh.thinf_na12mut= new_params['thinf_na12mut']
    currh.thi2_na12mut= new_params['thi2_na12mut']
    currh.thi1_na12mut= new_params['thi1_na12mut']
    currh.tha_na12mut= new_params['tha_na12mut']
    currh.vvs_na12mut= new_params['vvs_na12mut']
    currh.vvh_na12mut= new_params['vvh_na12mut']
    currh.vhalfs_na12mut= new_params['vhalfs_na12mut']
    currh.zetas_na12mut= new_params['zetas_na12mut']

    return

