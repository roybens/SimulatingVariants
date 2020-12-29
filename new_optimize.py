import numpy as np
import os
import subprocess
import shutil
import neuron as nrn
import bluepyopt as bpop

import struct
import time
import pandas as pd
import efel_ext
import matplotlib.pyplot as plt
import bluepyopt.deapext.algorithms as algo
from NeuroGPUFromPkl import run_params_with_pkl
from neuron import h
import optimize_na_ga_v3 as opt
import matplotlib.pyplot as plt
from scipy import optimize, stats
import SCN2A_nb_helper_actinact as nb
import genSimData_Na12ShortenTime as gsd
import time
import multiprocessing
from deap import algorithms, base, creator, tools
import random
import csv

exp_data_file = "./Data/NW_all_raw_data.csv"       # Experimental Data
param_names_file = "./Data/param_list.csv"    # List of parameter names

#########################
## Evaluator Functions ##
#########################

class neurogpu_evaluator(bpop.evaluators.Evaluator):
        def __init__(self, exp_data_file, params_file):
        """Constructor""" 
        self.params = init_params(params_file)
        self.objectives = [bpop.objectives.Objective('inact'),\
                           bpop.objectives.Objective('act'),\
                           bpop.objectives.Objective('recov'),\
                           bpop.objectives.Objective('tau0')
                           ]
        self.exp_data_map = read_all_raw_data(exp_data_file)


    def init_params(filepath):
        param_names, param_vals, param_min, param_max = np.loadtxt(filepath, delimiter = ',', unpack=True)
        param_list = []
        for i in range(len(param_names_array)):
            param_name = param_names_array[i]
            param_val = param_vals[i]
            min_bound = param_min[i]
            max_bound = param_max[i]
            param_names_list.append(bpop.parameters.Parameter(param_name, value=param_val, bounds=(min_bound, max_bound)))
        return param_list

    def evaluate(self, exp, mutant):
        real_data = self.exp_data_map[exp][mutant]
        return evaluate_with_lists(real_data)

    def evaluate_with_lists(self, param_values=[]):
        optimized_param_vals = self.run_model(param_values)
        for i in range(len(optimized_param_vals)):
            self.params.value = optimized_param_vals[i]


        '''
        scores = efel_ext.eval(target_volts, volts,times)   # Need to ask about this line
        return scores
        '''
    
    def run_model(self, param_values):
        '''
        '''
        pop, ga_stats, hof = opt.genetic_alg(param_values, ["inact", "act", "tau0"])
        print(hof)
        return list(hof[0])


############################
## Optimization Functions ##
############################
def genetic_alg(target_data, to_score=["inact", "act", "recov", "tau0"], pop_size=10, num_gens=50):
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
    global pool
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
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, nparams)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #use calc_rmse to score individuals
    toolbox.register("evaluate", calc_rmse)
    toolbox.register("mate", cx_two_point_copy)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    #allow multiprocessing
    
    #toolbox.register("map", pool.map)

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
            self.objectives[3].value = tau_rmse     # Update rmse of objectives
        else:
            if var == "inact":
                inds = global_target_data["inact sig inds"]
                squared_diffs = [(global_target_data[var][i]-sim_data[var][i])**2 for i in inds]
                inact_rmse = (sum(squared_diffs)/len(inds))**.5
                total_rmse = total_rmse + inact_rmse
                self.objectives[0].value = inact_rmse # Update rmse of objectives

            elif var == "act":
                inds = global_target_data["act sig inds"]
                squared_diffs = [(global_target_data[var][i]-sim_data[var][i])**2 for i in inds]
                act_rmse = (sum(squared_diffs)/len(inds))**.5
                total_rmse = total_rmse + act_rmse
                self.objectives[1].value = act_rmse # Update rmse of objectives


            elif var == "recov":
                inds = global_target_data["recov sig inds"]
                squared_diffs = [(global_target_data[var][i]-sim_data[var][i])**2 for i in inds]
                recov_rmse = (sum(squared_diffs)/len(inds))**.5
                total_rmse = total_rmse + recov_rmse
                self.objectives[2].value = recov_rmse # Update rmse of objectives


            else:
                print("cannot calc mse of {}".format(var))
                break
    print("rmse:{}".format(total_rmse))
    return (1/total_rmse,)

######################
## Helper Functions ##
######################
def read_all_raw_data(raw_data):
    '''
    Reads data in from CSV. 
    ---
    Return real_data: dictionary of experiments, each experiment is a 
    dictionary of mutants with the activation, inactivation, tau, 
    and recovery data recorded for that mutant.
    '''
    if raw_data = "./Data/NW_all_raw_data.csv":
        indexing_list = [3,20,21, 34, 35, 37, 51]
    else:
        indexing_list = [3,14,15, 29, 30, 32, 36]
        
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
        for i in range(indexing_list[0],indexing_list[1]):
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
        for i in range(indexing_list[2],indexing_list[3]):
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
        tau = float(lines[indexing_list[4]][col+1])
        mutant_data["tau0"] = tau
        
        #get recovery data
        recov_data = []
        times = []
        for i in range(indexing_list[5],indexing_list[6]):
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


algo._evaluate_invalid_fitness = neurogpu_evaluator.my_evaluate_invalid_fitness
