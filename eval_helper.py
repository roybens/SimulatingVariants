import generalized_genSim_shorten_time as ggsd

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
    if mutant_line == []:
        raise NameError('Invalid mutant name, or mutant is not yet in CSV database')
    protocols_dict = {}
    protocols_dict['dv_half_act'] = mutant_line[1]
    protocols_dict['gv_slope'] = mutant_line[2]
    protocols_dict['dv_half_ssi'] = mutant_line[3]
    protocols_dict['ssi_slope'] = mutant_line[4]
    protocols_dict['tau_fast'] = mutant_line[5]
    protocols_dict['tau_slow'] = mutant_line[6]
    protocols_dict['percent_fast'] = mutant_line[7]
    protocols_dict['udb20'] = mutant_line[8]
    protocols_dict['tau0'] = mutant_line[9]
    protocols_dict['ramp'] = mutant_line[10]
    protocols_dict['persistent'] = mutant_line[11]

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

def change_params(new_params, scaled=True):
    '''
    Change params on Na12mut channel in NEURON.
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

def make_params_dict(params_list):
    '''
    Make a dictionary of 24 parameters out of the raw values
    in PARAMS_LIST.
    ---
    params_list: list of raw parameter values, unscaled to be between 0 and 1
    '''
    params_dict = {
        'Ena_na12mut': params_list[0],
        'Rd_na12mut': params_list[1],
        'Rg_na12mut': params_list[2],
        'Rb_na12mut': params_list[3],
        'Ra_na12mut': params_list[4],
        'a0s_na12mut': params_list[5],
        'gms_na12mut': params_list[6],
        'hmin_na12mut': params_list[7],
        'mmin_na12mut': params_list[8],
        'qinf_na12mut': params_list[9],
        'q10_na12mut': params_list[10],
        'qg_na12mut': params_list[11],
        'qd_na12mut': params_list[12],
        'qa_na12mut': params_list[13],
        'smax_na12mut': params_list[14],
        'sh_na12mut': params_list[15],
        'thinf_na12mut': params_list[16],
        'thi2_na12mut': params_list[17],
        'thi1_na12mut': params_list[18],
        'tha_na12mut': params_list[19],
        'vvs_na12mut': params_list[20],
        'vvh_na12mut': params_list[21],
        'vhalfs_na12mut': params_list[22],
        'zetas_na12mut': params_list[23]
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
    bsae_value = {
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
    Change params on Na12mut channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    # params_orig = [0.02,7.2,7,0.4,0.124,0.03,-30,-45,-45,-45,0.01,2]
    #scale params up
    #new_params = scale_params_dict(False, new_params_dict)
    #get NEURON h
    currh = ggsd.activationNa12("geth")
    #change values of params
    #print(new_params)
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

