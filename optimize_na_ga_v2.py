"""
Sodium Channel Optimizer
--------------------
Bender Lab
____________________

Fits a NEURON mod file Sodium channel model to real data
or to an ideal Boltzmann model.

"""

from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats
import SCN2A_nb_helper_actinact as nb
import genSimData_Na12 as gsd
import time
import multiprocessing
from deap import algorithms, base, creator, tools
import random
import csv


###############
## Read Data ##
###############

def read_all_raw_data():
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
    with open("./all_raw_data.csv", 'r') as csv_file:
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

def get_mutant_list(exp, real_data=None):
    '''
    Generate list of mutant names for an experiment.
    ---
    Param exp: name of experiment
    Param real_data: dictionary of real data if available

    Return names: list of mutants in experiment
    '''
    #if data dict available, just use keys
    if real_data is not None:
        return list(real_data[exp].keys())

    #read raw data file to get mutants
    lines = []
    with open("./all_raw_data.csv", 'r') as csv_file:
        lines = [line.split(",") for line in csv_file]
    experiments = lines[0]
    mutants = lines[1]
    names = []
    for m in range(int((len(mutants)-1)/4)):
        col = m*4+1
        if exp == experiments[col]:
            names.append(mutants[col])
    return names

####################
## Simulated Data ##
####################

def gen_sim_data():
    '''
    Generate simulated data using the current NEURON state. Returns dictionary
    with activation, inactivation, tau, and recovery data.
    ---
    Return sim_data: dictionary of simulated data
    '''
    sim_data = {}

    #simulate activation
    act, act_sweeps, act_i = gsd.activationNa12("genActivation")
    sim_data["act"] = act.to_python()
    sim_data["act sweeps"] = act_sweeps.tolist()

    #calculate taus from inactivation
    taus, tau_sweeps, tau0 = gsd.find_tau_inact(act_i)
    sim_data["taus"] = taus
    sim_data["tau sweeps"] = tau_sweeps
    sim_data["tau0"] = tau0

    #simulate inactivation
    inact, inact_sweeps,inact_i = gsd.inactivationNa12("genInactivation")
    sim_data["inact"] = inact.to_python()
    sim_data["inact sweeps"] = inact_sweeps.tolist()

    #simulate recovery
    recov, recov_times = gsd.recInactTauNa12("genRecInact")
    sim_data["recov"] = recov
    sim_data["recov times"] = recov_times
    return sim_data

def scale_params(down, params):
    '''
    Scale parameters between 0 and 1.
    ---
    Param down: boolean to determine whether to scale down or up
    Param params: list of param values to scale

    Return: list of scaled param values
    '''
    #values to scale by
    scale_by = [0.02,7.2,7,0.4,0.124,0.003,-30,-85,-45,-85,0.001,2]
    #variable type (k = kinetic, v = voltage)
    types = ['k','k','k','k','k','k','v','v','v','v','k','k']

    bounds = []
    for i in range(len(scale_by)):
        val = scale_by[i]
        val_type = types[i]
        if val_type == 'k': #scale kinetic param
            bounds.append((val/25, val*5))
        elif val_type == 'v': #scale voltage param
            bounds.append((val-20, val+20))

    if down:
        return [(params[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(params))]
    return [params[i]*(bounds[i][1]-bounds[i][0]) + bounds[i][0] for i in range(len(params))]


def change_params(new_params_scaled):
    '''
    Change params on Na12mut channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    # params_orig = [0.02,7.2,7,0.4,0.124,0.03,-30,-45,-45,-45,0.01,2]
    #scale params up
    new_params = scale_params(False, new_params_scaled)
    #get NEURON h
    currh = gsd.activationNa12("geth")
    #change values of params
    currh.mmin_na12mut = new_params[0]
    currh.qa_na12mut = new_params[1]
    currh.qinf_na12mut = new_params[2]
    currh.Ra_na12mut = new_params[3]
    currh.Rb_na12mut = new_params[4]
    currh.Rd_na12mut = new_params[5]
    currh.tha_na12mut = new_params[6]
    currh.thi1_na12mut = new_params[7]
    currh.thinf_na12mut = new_params[8]
    currh.thi2_na12mut = new_params[9]
    currh.Rg_na12mut = new_params[10]
    currh.q10_na12mut = new_params[11]
    return

def scale_params_dict(down, params_dict):
    '''
    Scale parameters between 0 and 1.
    ---
    Param down: boolean to determine whether to scale down or up
    Param params: list of param values to scale

    Return: list of scaled param values
    '''
    #values to scale by
    scale_by = {
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


    bounds = {}
    for k, v in scale_by.items():
        val_type = types[k]
        if val_type == 'md': #scale kinetic param
            bounds[k] = (v/25, v*5)
        elif val_type == 'p': #scale voltage param
            bounds[k] = (v-20, v+20)
        else:
            bounds[k]= (0,1)

    if down:
        return [(v-bounds[k][0])/(bounds[k][1]-bounds[k][0]) for k,v in params_dict.items()]

    new_params = {}
    for  k,v  in params_dict.items():
        new_params[k]= v*(bounds[k][1]-bounds[k][0]) + bounds[k][0]
    return new_params



def change_params_dict(new_params_dict):
    '''
    Change params on Na12mut channel in NEURON.
    ---
    Param new_params_scaled: list of scaled param values
    '''
    # params_orig = [0.02,7.2,7,0.4,0.124,0.03,-30,-45,-45,-45,0.01,2]
    #scale params up
    new_params = scale_params_dict(False, new_params_dict)
    #get NEURON h
    currh = gsd.activationNa12("geth")
    #change values of params
    print(new_params)
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


##################
## Optimization ##
##################

def genetic_alg(target_data, to_score=["inact", "act", "recov", "tau0"], pop_size=300, num_gens=50):
    '''
    Runs DEAP genetic algorithm to optimize parameters of channel such that simulated data fits real data.
    ---
    Param target_data: data to fit
    Param to_score: list of simulations to run
    Param pop_size: size of population
    Param num_gens: number of generations

    Return pop: population at end of algorithm
    Return ga_stats: statistics of algorithm run
    Return hof: hall of fame object containing best individual (ie. best parameters)
    '''
    #set global variables for caculating error
    global global_target_data
    global global_to_score
    global_target_data = target_data
    global_to_score = to_score

    #Set goal to maximize rmse (which has been inverted)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #make "individual" an array of parameters
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    #randomly selected scaled param values between 0 and 1
    toolbox.register("attr_bool", random.uniform, 0, 1)
    #create individials as array of randomly selected scaled param values
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 12)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #use calc_rmse to score individuals
    toolbox.register("evaluate", calc_rmse)
    toolbox.register("mate", cx_two_point_copy)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    #allow multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=pop_size)
    #store best individual
    hof = tools.HallOfFame(1, similar=np.array_equal)

    #record statistics
    ga_stats = tools.Statistics(lambda ind: ind.fitness.values)
    ga_stats.register("avg", np.mean)
    ga_stats.register("std", np.std)
    ga_stats.register("min", np.min)
    ga_stats.register("max", np.max)

    #run DEAP algorithm
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_gens, stats=ga_stats,
                        halloffame=hof)
    return pop, ga_stats, hof

def calc_rmse(ind):
    '''
    Score individual using rmse.
    ---
    Param ind: DEAP individual object to score (essentially a list of param values)

    Return: tuple containing inverted rmse score (due to maximization)
    '''
    print(list(ind))
    #change params then simulate data
    change_params(ind)
    try:
        sim_data = gen_sim_data()
    except ZeroDivisionError: #catch error to prevent bad individuals from halting run
        print("ZeroDivisionError when generating sim_data, returned infinity.")

    total_rmse = 0
    #score only desired simulations at desired indicies
    for var in global_to_score:
        if var == "tau0":
            tau_rmse = ((global_target_data["tau0"]-sim_data["tau0"])**2)**.5
            total_rmse = total_rmse + tau_rmse
        else:
            if var == "inact":
                inds = global_target_data["inact sig inds"]
                squared_diffs = [(global_target_data[var][i]-sim_data[var][i])**2 for i in inds]
                inact_rmse = (sum(squared_diffs)/len(inds))**.5
                total_rmse = total_rmse + inact_rmse
            elif var == "act":
                inds = global_target_data["act sig inds"]
                squared_diffs = [(global_target_data[var][i]-sim_data[var][i])**2 for i in inds]
                act_rmse = (sum(squared_diffs)/len(inds))**.5
                total_rmse = total_rmse + act_rmse
            elif var == "recov":
                inds = global_target_data["recov sig inds"]
                squared_diffs = [(global_target_data[var][i]-sim_data[var][i])**2 for i in inds]
                recov_rmse = (sum(squared_diffs)/len(inds))**.5
                total_rmse = total_rmse + recov_rmse
            else:
                print("cannot calc mse of {}".format(var))
                break
    #normalize inact, act, recov, add voltage comparision. Keep log of all component for optimization, write components to file. Add more weight to recrovery, why is i5t not loking good.
    print("rmse:{}".format(total_rmse))
    return (1/total_rmse,)

def cx_two_point_copy(ind1, ind2):
    '''
    Funtion for mating individuals, copied from DEAP website.
    ---
    Params ind1,ind2: individuals to mate
    Return: mated individuals
    '''
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

def gen_boltz_and_opt(v05act=-15, slopeact=0.1, v05inact=-50, slopeinact=-0.1):
    '''
    Optimize params to fit an ideal Boltzman curve.
    ---
    Param v05act: desired V0.5 value for activation curve
    Param slopeact: desired slope of activation curve
    Param v05inact: desired V0.5 value for inactivation curve
    Param slopeinact: desired slope of inactivation curve

    Return: list of optimized param values
    '''
    #generate boltzmann data
    boltz_data = {}
    inact_sweeps, inact, act_sweeps, act = nb.gen_act_inact(v05act, slopeact, v05inact, slopeinact)
    boltz_data["inact"] = inact.tolist()
    boltz_data["inact sweeps"] = inact_sweeps.tolist()
    boltz_data["act"] = act.tolist()
    boltz_data["act sweeps"] = act_sweeps.tolist()
    boltz_data['inact sig inds'] = [i for i in range(0, len(inact))]
    boltz_data['act sig inds'] = [i for i in range(0, len(act))]

    #run genetic algorithm
    pop, ga_stats, hof = genetic_alg(boltz_data)
    print(hof)
    return list(hof[0])

def gen_real_and_opt(exp, mutant):
    '''
    Optimize params to fit real data.
    ---
    Param exp: name of experiment
    Param mutant: name of mutant

    Return: list of optimized param values
    '''
    real_data_map = read_all_raw_data()
    real_data = real_data_map[exp][mutant]
    pop, ga_stats, hof = genetic_alg(real_data, ["inact", "act", "tau0"])
    print(hof)
    return list(hof[0])


##############
## Plotting ##
##############

def fit_sigmoid(x, a, b):
    '''
    Fit a sigmoid curve to the array of datapoints.
    '''
    return 1.0 / (1.0+np.exp(-a*(x-b)))

def fit_exp(x, a, b, c):
    '''
    Fit an exponential curve to an array of datapoints.
    '''
    return a*np.exp(-b*x)+c

def gen_curves(data, names):
    '''
    Plot inactivation, activation, and recovery curves separately for each data set.
    ---
    Param data: list of data dictionaries
    Param names: list of names for data
    '''
    #plot inactivation
    for i in range(len(data)):
        data_pts = data[i]["inact"]
        sweeps = data[i]["inact sweeps"]
        #fit sigmoid curve to data
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        plt.scatter(sweeps, data_pts)
        plt.plot(even_xs, curve, label=names[i])
    plt.legend()
    plt.xlabel('Voltage')
    plt.ylabel('Fraction Inactivated')
    plt.title("Inactivation Curve")
    plt.show()

    #plot activation
    for i in range(len(data)):
        data_pts = data[i]["act"]
        sweeps = data[i]["act sweeps"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        plt.scatter(sweeps, data_pts)
        plt.plot(even_xs, curve, label=names[i])
    plt.legend()
    plt.xlabel('Voltage')
    plt.ylabel('Fraction Activated')
    plt.title("Activation Curve")
    plt.show()

    #plot recovery
    for i in range(len(data)):
        data_pts = data[i]["recov"]
        times = data[i]["recov times"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, times, data_pts, p0=[-.120, data_pts[0]], maxfev=5000)
        even_xs = np.linspace(times[0], times[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        plt.scatter(np.log(times), data_pts, label=names[i])
        plt.plot(even_xs, curve, label=names[i])

    plt.legend()
    plt.xlabel('Log(Time)')
    plt.ylabel('Fractional Recovery')
    plt.title("Recovery from Inactivation")
    plt.show()

def gen_curve_given_params(params):
    '''
    Generate curves given list of params.
    ---
    Params params: list of desired params
    '''
    change_params(params)
    sim_data = gen_sim_data()
    gen_curves([sim_data], ["sim data"])

def gen_figure_given_params(mutant, exp, params, target_data, save=True, file_name=None):
    '''
    Generate figure including all curves and tau value for a mutant.
    ---
    Param mutant: name of mutant
    Param exp: name of experiment
    Param params: list of params to use
    Param target_data: data to plot for comparison
    Param save: boolean for saving figure
    Param file_name: desired file name to save as
    '''
    #set-up figure
    plt.close()
    fig, axs = plt.subplots(2, figsize=(10,10))
    fig.suptitle("Mutant: {} \n Experiment: {}".format(mutant, exp))
    change_params(params)
    sim_data = gen_sim_data()
    data = [target_data, sim_data]
    names = ["experimental", "simulated"]
    max_calls = 500

    #plot inactivation and activation curves on same axis

    axs[0].set_xlabel('Voltage')
    axs[0].set_ylabel('Fraction In/activated')
    axs[0].set_title("Inactivation and Activation Curves")
    for i in range(len(data)):
        data_pts = data[i]["inact"]
        sweeps = data[i]["inact sweeps"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=max_calls)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        
        if names[i] == "experimental":
            axs[0].scatter(sweeps, data_pts, color='black',marker='s')
            axs[0].plot(even_xs, curve, color='black', label=names[i]+" inactivation")
        else:
            axs[0].scatter(sweeps, data_pts, color='red',marker='s')
            axs[0].plot(even_xs, curve, color='red', label=names[i]+" inactivation")
    for i in range(len(data)):
        data_pts = data[i]["act"]
        sweeps = data[i]["act sweeps"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, sweeps, data_pts, p0=[-.120, data_pts[0]], maxfev=max_calls)
        even_xs = np.linspace(sweeps[0], sweeps[len(sweeps)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        if names[i] == "experimental":
            axs[0].scatter(sweeps, data_pts, color='black',marker='o')
            axs[0].plot(even_xs, curve, color='black', label=names[i]+" activation")
        else:
            axs[0].scatter(sweeps, data_pts, color='red',marker='o')
            axs[0].plot(even_xs, curve, color='red', label=names[i]+" activation")
    axs[0].legend()

    #plot recovery curves
    axs[1].set_xlabel('Log(Time)')
    axs[1].set_ylabel('Fractional Recovery')
    axs[1].set_title("Recovery from Inactivation")
    for i in range(len(data)):
        data_pts = data[i]["recov"]
        times = data[i]["recov times"]
        popt, pcov = optimize.curve_fit(fit_sigmoid, times, data_pts, p0=[-.120, data_pts[0]], maxfev=max_calls)
        even_xs = np.linspace(times[0], times[len(times)-1], 100)
        curve = fit_sigmoid(even_xs, *popt)
        axs[1].semilogx(even_xs, curve, c="r",label=names[i]+" recovery")
        axs[1].scatter(np.log(times), data_pts, label=names[i], color='black',marker='o')
        
    axs[1].legend()

    #add text containing tau information
    fig.text(.5, .92, "\n Target tau: {}, Sim tau: {}".format(target_data['tau0'], sim_data['tau0']), ha='center')
    plt.show()

    #save figure
    if save:
        if file_name is None:
            file_name = "{}_{}_plots".format(exp, mutant).replace(" ", "_")
        fig.savefig("./curves/"+file_name+'.eps')
        fig.savefig("./curves/"+file_name+'.pdf')

def plot_real_opt(exp, mutant, params, save=False):
    '''
    Plot real and optimized data in figure.
    ---
    Param exp: name of experiment
    Param mutant:  name of mutant
    Param params: list of param values
    Param save: boolean for saving
    '''
    real_data_map = read_all_raw_data()
    real_data = real_data_map[exp][mutant]
    gen_figure_given_params(mutant, exp, params, real_data, save=save)

##############
## Pipeline ##
##############

def make_params_dict(exp, name, params, scale=True):
    '''
    Convert list of params into dictionary of params.
    ---
    Param exp: name of experiment
    Param name: name of mutant
    Param params: list of param values
    Param scale: whether to scale values up before saving

    Return params_dict: dictionary of params
    '''
    if scale:
        params = scale_params(False, params)

    params_dict = {}
    params_dict["exp"] = exp
    params_dict["name"] = name
    params_dict["mmin"] = params[0]
    params_dict["qa"] = params[1]
    params_dict["qinf"] = params[2]
    params_dict["Ra"] = params[3]
    params_dict["Rb"] = params[4]
    params_dict["Rd"] = params[5]
    params_dict["tha"] = params[6]
    params_dict["thi1"] = params[7]
    params_dict["thinf"] = params[8]
    params_dict["thi2"] = params[9]
    params_dict["Rg"] = params[10]
    params_dict["q10"] = params[11]
    return params_dict

def save_dict(params_dict, name):
    '''
    Save params dictionary as CSV.
    ---
    Param params_dict: dictionary to save
    Param name: file name to save under
    '''
    w = csv.writer(open("./param_dicts/{}.csv".format(name.replace(" ", "_")), "w"))
    for key, val in params_dict.items():
        w.writerow([key, val])

def opt_na_pipeline(exp, mutant=None):
    '''
    Optimization pipeline.
    ---
    Param exp: name of experiment
    Param mutant: name of mutant
    '''
    #if no mutant given, run on all mutants in experiment
    print("Gett Mutant List")
    if mutant == None:
        mutants = get_mutant_list(exp)
        print(mutants)
    else:
        mutants = [mutant]

    for mut in mutants:
        print("Optimizing: {}".format(mut))
        t0 = time.time()
        opt_params = gen_real_and_opt(exp, mut)
        plot_real_opt(exp, mut, opt_params, save=True)
        t1 = time.time()
        print("runtime: {}".format(t1-t0))
        opt_dict = make_params_dict(exp, mut, opt_params)
        save_dict(opt_dict, exp+mut+"_params_new".replace(" ", "_"))

##########
## Main ##
##########

def main():
    '''
    Main method.
    '''
    refits = [('M1879 T and R1626Q', 'NaV12 adult R1626Q'),
              ('M1879 T and R1626Q', 'NaV12 adult M1879T')]

    for exp, mut in refits:
        opt_na_pipeline(exp, mut)


if __name__ == '__main__':
   main()
