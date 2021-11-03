import generalized_genSim_tel_aviv as ggsd

scale_voltage = 30
scale_fact = 7.5

def read_mutant_protocols(mutant_protocol_csv):
    '''
    Reads data for a single MUTANT from a csv of mutant protocols.
    Returns a dictionary with all the relevant protocols for that 
    MUTANT.
    '''
    lines = []
    with open(mutant_protocol_csv, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]

    protocols_dict = {}
    protocols_dict['iv'] = lines[1][3]
    protocols_dict['dv_half_act'] = lines[2][3]
    protocols_dict['dv_half_ssi'] = lines[3][3]
    protocols_dict['tau'] = lines[4][3]
    protocols_dict['gv_slope'] = lines[5][3]
    protocols_dict['ssi_slope'] = lines[6][3]
    protocols_dict['persistent'] = lines[7][3]

    return protocols_dict

def read_all_raw_data_SCN8A(raw_data):
    '''
    Reads data in from CSV.
    ---
    Return real_data: dictionary of experiments, each experiment is a
    dictionary of mutants with the activation, inactivation, tau,
    and recovery data recorded for that mutant.
    '''
    #open file
    print("Start:")
    lines = []
    with open(raw_data, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]

    #get all experiment names and make dictionary
    experiments = lines[0]
    real_data = {}
    for e in experiments:
        real_data[e] = {}

    #get all mutants
    mutants = lines[1]
    print("Mutants", mutants)
    for m in range(1):
        col = 1 #select column containing mean data
        name = mutants[col]
        exp = experiments[col]
        print("name")
        print(name, exp)
        unique_name = "{} ({})".format(name, exp)
        mutant_data = {}
        mutant_data["unique name"] = unique_name

        #get activation data
        act_curve = []
        sweeps_act = [] #stim voltages
        for i in range(3,14):
            sweeps_act.insert(i,float(lines[i][col]))
            act_curve.insert(i, float(lines[i][col+1]))
        mutant_data["act"] = act_curve
        mutant_data["act sweeps"] = sweeps_act
        act_sig_indices = []
        #select significant indicies
        for ind in range(len(act_curve)):
            curr_frac = act_curve[ind]
            if (abs(1-curr_frac)>0.05 and abs(curr_frac)>0.05):
                act_sig_indices.append(ind)
        mutant_data["act sig inds"] = act_sig_indices

        #get inactivation data
        inact_curve = []
        sweeps_inact = []
        for i in range(15,29):
            sweeps_inact.insert(i,float(lines[i][col]))
            inact_curve.insert(i, float(lines[i][col+1]))
        mutant_data["inact"] = inact_curve
        mutant_data["inact sweeps"] = sweeps_inact
        inact_sig_indices = []
        for ind in range(len(inact_curve)):
            curr_frac = inact_curve[ind]
            if abs(1-curr_frac)>0.05 and abs(curr_frac)>0.05:
                inact_sig_indices.append(ind)
        mutant_data["inact sig inds"] = inact_sig_indices

        #get tau value
        tau = float(lines[30][col+1])
        mutant_data["tau0"] = tau

        #get recovery data
        recov_data = []
        times = []
        for i in range(32,36):
            times.insert(i,float(lines[i][col]))
            recov_data.insert(i, float(lines[i][col+1]))
        mutant_data["recov"] = recov_data
        mutant_data["recov times"] = times
        print("Test:")
        print(mutant_data)
        #select all indicies as significant since unsure how to determine
        mutant_data["recov sig inds"] = [i for i in range(len(recov_data))]
        real_data[exp][name] = mutant_data

    #remove extra keys
    for key in [key for key in real_data if real_data[key] == {}]: del real_data[key]
    return real_data

def read_all_raw_data(raw_data):
    '''
    Reads data in from CSV. 
    ---
    Return real_data: dictionary of experiments, each experiment is a 
    dictionary of mutants with the activation, inactivation, tau, 
    and recovery data recorded for that mutant.
    '''
    #open file
    lines = []
    with open(raw_data, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]
        
    #get all experiment names and make dictionary
    experiments = lines[0]
    real_data = {}
    for e in experiments:
        real_data[e] = {}
        
    #get all mutants
    mutants = lines[1]
    for m in range(int((len(mutants)-1)/4)):
        col = m*4+1 #select column containing mean data
        name = mutants[col]
        exp = experiments[col]
        unique_name = "{} ({})".format(name, exp)
        mutant_data = {}
        mutant_data["unique name"] = unique_name
        
        #get activation data
        act_curve = []
        sweeps_act = [] #stim voltages
        for i in range(3,20):
            sweeps_act.insert(i,float(lines[i][col]))
            act_curve.insert(i, float(lines[i][col+1]))
        mutant_data["act"] = act_curve
        mutant_data["act sweeps"] = sweeps_act
        act_sig_indices = []
        #select significant indicies
        for ind in range(len(act_curve)):
            curr_frac = act_curve[ind]
            if (abs(1-curr_frac)>0.05 and abs(curr_frac)>0.05):
                act_sig_indices.append(ind)
        mutant_data["act sig inds"] = act_sig_indices
        
        #get inactivation data
        inact_curve = []
        sweeps_inact = []
        for i in range(21,34):
            sweeps_inact.insert(i,float(lines[i][col]))
            inact_curve.insert(i, float(lines[i][col+1]))
        mutant_data["inact"] = inact_curve
        mutant_data["inact sweeps"] = sweeps_inact
        inact_sig_indices = []
        for ind in range(len(inact_curve)):
            curr_frac = inact_curve[ind]
            if abs(1-curr_frac)>0.05 and abs(curr_frac)>0.05:
                inact_sig_indices.append(ind)
        mutant_data["inact sig inds"] = inact_sig_indices
        
        #get tau value
        tau = float(lines[35][col+1])
        mutant_data["tau0"] = tau
        
        #get recovery data
        recov_data = []
        times = []
        for i in range(37,51):
            times.insert(i,float(lines[i][col]))
            recov_data.insert(i, float(lines[i][col+1]))
        mutant_data["recov"] = recov_data
        mutant_data["recov times"] = times
        #select all indicies as significant since unsure how to determine
        mutant_data["recov sig inds"] = [i for i in range(len(recov_data))]
        real_data[exp][name] = mutant_data

    #remove extra keys
    for key in [key for key in real_data if real_data[key] == {}]: del real_data[key] 
    return real_data

def read_HMM_parameters(csv_data_path = './HMM_params.csv'):
    '''
    Reads data for a csv file of HMM parameters.
    Returns a dictionary with all the relevant parameters. Each mapped
        value in the dictionary is a list with 3 elements, each being the 
        value, half value, and double value
    '''
    lines = []
    with open(csv_data_path, 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]

    #Each line[0] except the first should contain the name of the param.
    params_dict = {}
    for line in lines:
        params_dict[line[0]] = line[1:4]

    return params_dict

def change_params(new_params, scaled=True):
    '''
    Change params on na16 channel in NEURON.
    ---
    Param new_params_scaled: list of param values
        scaled: whether the parameters are scaled to be between 0 and 1
    '''
    if scaled:
        new_param_dict = scale_params_dict(False, new_params)
    else:
        new_param_dict = make_params_dict(new_params)
    change_params_dict(new_param_dict)
    return

def make_params_dict(params_list, is_HMM = False):
    '''
    Make a dictionary of 24 parameters out of the raw values
    in PARAMS_LIST.
    ---
    params_list: list of raw parameter values, unscaled to be between 0 and 1
    '''
    params_dict = {
        'Ena_na16': params_list[0],
        'Rd_na16': params_list[1],
        'Rg_na16': params_list[2],
        'Rb_na16': params_list[3],
        'Ra_na16': params_list[4],
        'a0s_na16': params_list[5],
        'gms_na16': params_list[6],
        'hmin_na16': params_list[7],
        'mmin_na16': params_list[8],
        'qinf_na16': params_list[9],
        'q10_na16': params_list[10],
        'qg_na16': params_list[11],
        'qd_na16': params_list[12],
        'qa_na16': params_list[13],
        'smax_na16': params_list[14],
        'sh_na16': params_list[15],
        'thinf_na16': params_list[16],
        'thi2_na16': params_list[17],
        'thi1_na16': params_list[18],
        'tha_na16': params_list[19],
        'vvs_na16': params_list[20],
        'vvh_na16': params_list[21],
        'vhalfs_na16': params_list[22],
        'zetas_na16': params_list[23]
    }
    return params_dict

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
        'Ena_na16': 55,
        'Rd_na16': .03,
        'Rg_na16': .01,
        'Rb_na16': .124,
        'Ra_na16': 0.4,
        'a0s_na16': 0.0003,
        'gms_na16': .2,
        'hmin_na16': .01,
        'mmin_na16': .02,
        'qinf_na16': 7,
        'q10_na16': 2,
        'qg_na16': 1.5,
        'qd_na16': .5,
        'qa_na16': 7.2,
        'smax_na16': 10,
        'sh_na16': 8,
        'thinf_na16': -55,
        'thi2_na16': -45,
        'thi1_na16': -45,
        'tha_na16': -35,
        'vvs_na16': 2,
        'vvh_na16': -58,
        'vhalfs_na16': -60,
        'zetas_na16': 12
    }
    types = {
        'Ena_na16': 'p',
        'Rd_na16': 'p',
        'Rg_na16': 'p',
        'Rb_na16': 'p',
        'Ra_na16': 'p',
        'a0s_na16': 'md',
        'gms_na16': 'p',
        'hmin_na16': 'p',
        'mmin_na16': 'p',
        'qinf_na16': 'md',
        'q10_na16': 'p',
        'qg_na16': 'md',
        'qd_na16': 'md',
        'qa_na16': 'md',
        'smax_na16': 'p',
        'sh_na16': 'p',
        'thinf_na16': 'p',
        'thi2_na16': 'p',
        'thi1_na16': 'p',
        'tha_na16': 'p',
        'vvs_na16': 'p',
        'vvh_na16': 'p',
        'vhalfs_na16': 'p',
        'zetas_na16': 'p'
    }
    inds = {
        'Ena_na16': 0,
        'Rd_na16': 1,
        'Rg_na16': 2,
        'Rb_na16': 3,
        'Ra_na16': 4,
        'a0s_na16': 5,
        'gms_na16': 6,
        'hmin_na16': 7,
        'mmin_na16': 8,
        'qinf_na16': 9,
        'q10_na16': 10,
        'qg_na16': 11,
        'qd_na16': 12,
        'qa_na16': 13,
        'smax_na16': 14,
        'sh_na16': 15,
        'thinf_na16': 16,
        'thi2_na16': 17,
        'thi1_na16': 18,
        'tha_na16': 19,
        'vvs_na16': 20,
        'vvh_na16': 21,
        'vhalfs_na16': 22,
        'zetas_na16': 23
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

    return new_params

def change_params_dict(new_params):
    '''
    Change params on na16 channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    # params_orig = [0.02,7.2,7,0.4,0.124,0.03,-30,-45,-45,-45,0.01,2]
    #get NEURON h
    currh = ggsd.Activation().h
    #change values of params
    currh.Rd_na16= new_params['Rd_na16']
    currh.Rg_na16= new_params['Rg_na16']
    currh.Rb_na16= new_params['Rb_na16']
    currh.Ra_na16= new_params['Ra_na16']
    currh.a0s_na16= new_params['a0s_na16']
    currh.gms_na16= new_params['gms_na16']
    currh.hmin_na16= new_params['hmin_na16']
    currh.mmin_na16= new_params['mmin_na16']
    currh.qinf_na16= new_params['qinf_na16']
    currh.q10_na16= new_params['q10_na16']
    currh.qg_na16= new_params['qg_na16']
    currh.qd_na16= new_params['qd_na16']
    currh.qa_na16= new_params['qa_na16']
    currh.smax_na16= new_params['smax_na16']
    currh.sh_na16= new_params['sh_na16']
    currh.thinf_na16= new_params['thinf_na16']
    currh.thi2_na16= new_params['thi2_na16']
    currh.thi1_na16= new_params['thi1_na16']
    currh.tha_na16= new_params['tha_na16']
    currh.vvs_na16= new_params['vvs_na16']
    currh.vvh_na16= new_params['vvh_na16']
    currh.vhalfs_na16= new_params['vhalfs_na16']
    currh.zetas_na16= new_params['zetas_na16']
    return

def gen_sim_data():
    '''
    Generate simulated data using the current NEURON state. Returns dictionary
    with activation, inactivation, tau, and recovery data.
    ---
    Return sim_data: dictionary of simulated data
    '''
    sim_data = {}

    #simulate activation
    act, act_sweeps, act_i = ggsd.activationNa12("genActivation")
    sim_data["act"] = act.to_python()
    sim_data["act sweeps"] = act_sweeps.tolist()

    #simulate inactivation
    inact, inact_sweeps,inact_i = ggsd.inactivationNa12("genInactivation")
    sim_data["inact"] = inact.to_python()
    sim_data["inact sweeps"] = inact_sweeps.tolist()

    #calculate taus from inactivation
    taus, tau_sweeps, tau0 = ggsd.find_tau_inact(inact_i)
    sim_data["taus"] = taus
    sim_data["tau sweeps"] = tau_sweeps
    sim_data["tau0"] = tau0

    #simulate recovery
    recov, recov_times = ggsd.recInactTauNa12("genRecInact")
    sim_data["recov"] = recov
    sim_data["recov times"] = recov_times
    return sim_data

