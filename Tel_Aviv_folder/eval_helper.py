import generalized_genSim_tel_aviv as ggsd

scale_voltage = 30
scale_fact = 7.5

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
    protocols_dict['iv'] = mutant_line[1]
    protocols_dict['dv_half_act'] = mutant_line[2]
    protocols_dict['dv_half_ssi'] = mutant_line[3]
    protocols_dict['tau'] = mutant_line[4]
    protocols_dict['gv_slope'] = mutant_line[5]
    protocols_dict['ssi_slope'] = mutant_line[6]
    protocols_dict['persistent'] = mutant_line[7]

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

    if is_HMM:
        params_dict['C1C2b2'] = param_list[24]
        params_dict['C1C2v2'] = param_list[25]
        params_dict['C1C2k2'] = param_list[26]
        params_dict['C2C1b1'] = param_list[27]
        params_dict['C2C1v1'] = param_list[28]
        params_dict['C2C1k1'] = param_list[29]
        params_dict['C2C1b2'] = param_list[30]
        params_dict['C2C1v2'] = param_list[31]
        params_dict['C2C1k2'] = param_list[32]
        params_dict['C2O1b2'] = param_list[33]
        params_dict['C2O1v2'] = param_list[34]
        params_dict['C2O1k2'] = param_list[35]
        params_dict['O1C2b1'] = param_list[36]
        params_dict['O1C2v1'] = param_list[37]
        params_dict['O1C2k1'] = param_list[38]
        params_dict['O1C2b2'] = param_list[39]
        params_dict['O1C2v2'] = param_list[40]
        params_dict['O1C2k2'] = param_list[41]
        params_dict['O1I1b1'] = param_list[42]
        params_dict['O1I1v1'] = param_list[43]
        params_dict['O1I1k1'] = param_list[44]
        params_dict['O1I1b2'] = param_list[45]
        params_dict['O1I1v2'] = param_list[46]
        params_dict['O1I1k2'] = param_list[47]
        params_dict['I1O1b1'] = param_list[48]
        params_dict['I1O1v1'] = param_list[49]
        params_dict['I1O1k1'] = param_list[50]
        params_dict['I1C1b1'] = param_list[51]
        params_dict['I1C1v1'] = param_list[52]
        params_dict['I1C1k1'] = param_list[53]
        params_dict['C1I1b2'] = param_list[54]
        params_dict['C1I1v2'] = param_list[55]
        params_dict['C1I1k2'] = param_list[56]
        params_dict['I1I2b2'] = param_list[57]
        params_dict['I1I2v2'] = param_list[58]
        params_dict['I1I2k2'] = param_list[59]
        params_dict['I2I1b1'] = param_list[60]
        params_dict['I2I1v1'] = param_list[61]
        params_dict['I2I1k1'] = param_list[62]
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
    bsae_value = {
    'Ena_na16': 55,
    'Rd_na16': .03,
    'Rg_na16': .01,
    'Rb_na16': .124,
    'Ra_na16': 0.4,
    'a0s_na16': 0.0003,
    'gms_na16': .02,
    'hmin_na16': .01,
    'mmin_na16': .02,
    'qinf_na16': 7,
    'q10_na16': 2,
    'qg_na16': 1.5,
    'qd_na16': .5,
    'qa_na16': 7.2,
    'smax_na16': 10,
    'sh_na16': 8,
    'thinf_na16': -45,
    'thi2_na16': -45,
    'thi1_na16': -45,
    'tha_na16': -30,
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
    for k, v in bsae_value.items():
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
    Change params on na16 channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    # params_orig = [0.02,7.2,7,0.4,0.124,0.03,-30,-45,-45,-45,0.01,2]
    #scale params up
    #new_params = scale_params_dict(False, new_params_dict)
    #get NEURON h
    currh = ggsd.Activation().h
    #change values of params
    #print(new_params)
    currh.na16= new_params['Rd_na16']
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

def change_params_dict_gen(new_params):
    '''
    Change params on na16 channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    # params_orig = [0.02,7.2,7,0.4,0.124,0.03,-30,-45,-45,-45,0.01,2]
    #scale params up
    #new_params = scale_params_dict(False, new_params_dict)
    #get NEURON h
    currh = ggsd.Activation().h
    #change values of params
    #print(new_params)
    currh.Rd_na16= new_params['Rd']
    currh.Rg_na16= new_params['Rg']
    currh.Rb_na16= new_params['Rb']
    currh.Ra_na16= new_params['Ra']
    currh.a0s_na16= new_params['a0']
    currh.gms_na16= new_params['gms']
    currh.hmin_na16= new_params['hmin']
    currh.mmin_na16= new_params['mmin']
    currh.qinf_na16= new_params['qinf']
    currh.q10_na16= new_params['q10']
    currh.qg_na16= new_params['qg']
    currh.qd_na16= new_params['qd']
    currh.qa_na16= new_params['qa']
    currh.smax_na16= new_params['smax']
    currh.sh_na16= new_params['sh']
    currh.thinf_na16= new_params['thinf']
    currh.thi2_na16= new_params['thi2']
    currh.thi1_na16= new_params['thi1']
    currh.tha_na16= new_params['tha']
    currh.vvs_na16= new_params['vvs']
    currh.vvh_na16= new_params['vvh']
    currh.vhalfs_na16= new_params['vhalfs']
    currh.zetas_na16= new_params['zetas']
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

