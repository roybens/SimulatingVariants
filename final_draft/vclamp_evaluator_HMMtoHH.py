############################################
## Code for Hidden-Markov Model evaluator ##
########## Author: Michael Lam #############
############################################

import bluepyopt as bpop
import matplotlib.pyplot as plt
import numpy as np
import curve_fitting as cf
import eval_helper as eh
import \
    scoring_functions_relative as sf  # Change this to scoring_functions_relative or scoring_functions_exp to change scoring functions
from generate_simulation import *


class Vclamp_evaluator_HMM(bpop.evaluators.Evaluator):
    '''
    A class that holds a set of objectives and a set of parameters.
    
    self.params holds the names of each of the parameters to be evaluated along with 
    their bounds and values

    self.objectives holds a set of categories for which an error will be calculated 
    through the evaluate_with_lists function
    '''


    def __init__(self, params_file, mutant, channel_name_HMM, channel_name_HH, objective_names=[]):
        '''
        Constructor

        exp_data_file: a filepath to a csv containing the experimental data, NW style
        params_file: a filepath to a csv containing the names, starting values, and bounds
            of each parameter in the following format:

            parameter name | parameter value | lower bound | upper bound

                  sh       |         5       |      3      |      15
                  ...      |        ...      |     ...     |      ...

        mutant: name of the mutant

        '''
        self.channel_name_HMM = channel_name_HMM
        self.channel_name_HH = channel_name_HH
        self.objective_names = objective_names
        def init_params(filepath):
            '''
            Helper to initialize self.params with the parameter file from filepath

            filepath: filepath to csv containing parameter stats
            '''
            param_names_array = np.loadtxt(filepath, dtype=str, delimiter = ',', skiprows=1, usecols=(0), unpack=True, max_rows=24)
            param_vals, param_min, param_max = np.loadtxt(filepath, delimiter = ',', skiprows=1, usecols=(1, 2 ,3), unpack=True, max_rows=24)
            param_list = []
            for i in range(len(param_names_array)):
                param_name = param_names_array[i]
                param_val = param_vals[i]
                min_bound = param_min[i]
                max_bound = param_max[i]
                param_list.append(bpop.parameters.Parameter(param_name, value=param_val, bounds=(min_bound, max_bound)))
            return param_list

        self.wild_data = self.initialize_wild_data()
        self.params = init_params(params_file)
        self.mutant = mutant
        self.objectives = []
        for obj in objective_names:
            self.objectives.append(bpop.objectives.Objective(obj))

        self.protocols = eh.read_mutant_protocols('csv_files/mutant_protocols.csv', mutant)
        self.score_calculator = sf.Score_Function(self.protocols, self.wild_data, self.channel_name_HMM)

    def initialize_wild_data(self):
        '''
        Using the current channel's mod file, calculate and load all objective values into a dict.

        Arguments:
            None

        Returns:
            Dictionary of objective values for the wild channel
        '''
        
        wild_data = {}
        # Getting objective base values for HH model.
        is_HMM = False 
        # Create genSim objects
        act_obj = Activation_general(channel_name=self.channel_name_HH)
        tau0 = act_obj.get_Tau_0mV()
        act_obj = Activation_general(channel_name=self.channel_name_HH)
        peak_amp = act_obj.find_peak_amp([14, 33])
        time_to_peak = act_obj.find_time_to_peak([14, 33])
        gv_slope, v_half_act, top, bottom = cf.calc_act_obj(act_obj)
        inact_obj = Inactivation_general(channel_name=self.channel_name_HH)
        ssi_slope, v_half_inact, top, bottom = cf.calc_inact_obj(inact_obj)
        wild_data['v_half_act'] = v_half_act
        wild_data['gv_slope'] = gv_slope
        wild_data['v_half_ssi'] = v_half_inact
        wild_data['ssi_slope'] = ssi_slope
        wild_data['tau0'] = tau0
        wild_data['peak_amp'] = peak_amp
        wild_data['time_to_peak'] = time_to_peak
        self.wild_data = wild_data
        #print(wild_data)
        return wild_data


    def evaluate_with_lists(self, param_values=[]):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors
        '''
        return self.calc_all_rmse(param_values)
    

    def calc_all_rmse(self, param_values):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors

        '''
        assert len(param_values) == len(self.params), 'Parameter value list is not same length number of parameters' 
        act_obj = Activation_general(channel_name=self.channel_name_HMM)
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=act_obj)

        inact_obj = Inactivation_general(channel_name=self.channel_name_HMM)
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=inact_obj)
        score = self.score_calculator.total_rmse(act_obj, inact_obj, None, is_HMM=True, objectives=self.objective_names)
        return score
    

        
    def unit_test(self):
        inorm_vec, v_vec, all_is = ggsdHMM.Inactivation(channel_name=self.channel_name, step=5).genInactivation()
        plot_inact(self)
        plot_act(self)


    
