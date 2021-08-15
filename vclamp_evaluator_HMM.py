############################################
## Code for Hidden-Markov Model evaluator ##
########## Author: Michael Lam #############
############################################

import numpy as np
import bluepyopt as bpop
import eval_helper as eh
import scoring_functions_exp as sf
import curve_fitting as cf
import matplotlib.pyplot as plt
import generalized_genSim_shorten_time_HMM as ggsdHMM

class Vclamp_evaluator_HMM(bpop.evaluators.Evaluator):
    '''
    A class that holds a set of objectives and a set of parameters.
    
    self.params holds the names of each of the parameters to be evaluated along with 
    their bounds and values

    self.objectives holds a set of categories for which an error will be calculated 
    through the evaluate_with_lists function
    '''

    def __init__(self, params_file, mutant, channel_name, objective_names=['v_half_act', 'gv_slope', 'v_half_ssi', 'ssi_slope', 'tau_fast', 'tau_slow', 'percent_fast', 'udb20', 'tau0', 'ramp', 'persistent']):
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
        self.channel_name = channel_name
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
        
        self.objectives = []
        for obj in objective_names:
            self.objectives.append(bpop.objectives.Objective(obj))

        self.protocols = eh.read_mutant_protocols('mutant_protocols.csv', mutant)
        self.score_calculator = sf.Score_Function(self.protocols, self.wild_data, self.channel_name)
        

    def initialize_wild_data(self):
        '''
        Using the current channel's mod file, calculate and load all objective values into a dict.

        Arguments:
            None

        Returns:
            Dictionary of objective values for the wild channel
        '''
        wild_data = {}
        is_HMM = True   # This is an HMM model
        gv_slope, v_half_act, top, bottom = cf.calc_act_obj(self.channel_name, is_HMM=is_HMM)
        ssi_slope, v_half_inact, top, bottom, tau0 = cf.calc_inact_obj(self.channel_name, is_HMM=is_HMM)
        y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(self.channel_name, is_HMM=is_HMM)

        wild_data['v_half_act'] = v_half_act
        wild_data['gv_slope'] = gv_slope
        wild_data['v_half_ssi'] = v_half_inact
        wild_data['ssi_slope'] = ssi_slope
        wild_data['tau_fast'] = 1 / k_fast
        wild_data['tau_slow'] = 1 / k_slow
        wild_data['percent_fast'] = percent_fast
        wild_data['udb20'] = 0
        wild_data['tau0'] = tau0
        wild_data['ramp'] = 0
        wild_data['persistent'] = 0

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
        eh.change_params(param_values, scaled=False, is_HMM=True)
        #score_calculator = sf.Score_Function(self.protocols, self.wild_data, self.channel_name)
        return self.score_calculator.total_rmse(is_HMM=True, objectives=self.objective_names)



    def plot_data(self, param_values, mutant):
        '''
        Plot activation, inactivation, and recovery of the channel corresponding with the PARAM_VALUES overlaid on 
        that of the wild MUTANT channel.

        Arguments:
            param_values: list of floar parameter values in order

        Returns:
            None
        '''
        eh.change_params(param_values, scaled=False, is_HMM=True)
        plt.close()
        fig, axs = plt.subplots(3, figsize=(10,10))
        fig.suptitle("Mutant: {}".format(mutant))
    
        # Calculate wild baseline values
        param_dict_rel = eh.read_mutant_protocols('mutant_protocols.csv', mutant)
        v_half_act_exp = self.wild_data['v_half_act'] + float(self.protocols['dv_half_act']) /100
        gv_slope_exp = self.wild_data['gv_slope'] * float(self.protocols['gv_slope']) / 100
        v_half_ssi_exp = self.wild_data['v_half_ssi'] + float(self.protocols['dv_half_ssi'])
        ssi_slope_exp = self.wild_data['ssi_slope'] * float(self.protocols['ssi_slope']) / 100
        tau_fast_exp = self.wild_data['tau_fast'] * float(self.protocols['tau_fast']) / 100
        tau_slow_exp = self.wild_data['tau_slow'] * float(self.protocols['tau_slow']) / 100
        percent_fast_exp = self.wild_data['percent_fast'] * float(self.protocols['percent_fast']) / 100
        udb20_exp = 0
        tau0_exp = self.wild_data['tau0'] * float(self.protocols['tau0']) / 100

        ramp_exp = 0
        persistent_exp = 0


        # Inactivation curve
        inorm_vec, v_vec, all_is = ggsdHMM.Inactivation(channel_name=self.channel_name, step=5).genInactivation()
        inorm_array = np.array(inorm_vec)
        v_array = np.array(v_vec)

        ssi_slope, v_half, top, bottom, tau0 = cf.calc_inact_obj(self.channel_name, is_HMM=True)

        
        even_xs = np.linspace(v_array[0], v_array[len(v_array)-1], 100)
        curve = cf.boltzmann(even_xs, ssi_slope, v_half, top, bottom)
        axs[0].set_xlabel('Voltage (mV)')
        axs[0].set_ylabel('Fraction Inactivated')
        axs[0].set_title("Inactivation Curve")
        axs[0].scatter(v_array, inorm_array, color='red', marker='s', label='Optimized Inactivation')
        axs[0].plot(even_xs, curve, color='red', label="Fitted Inactivation")

        
        curve_exp = cf.boltzmann(even_xs, ssi_slope_exp, v_half_ssi_exp, top, bottom)
        axs[0].plot(even_xs, curve_exp, color='black', label='Inactivation experimental')
        axs[0].text(-120, 0.7, 'Slope (Optimized): ' + str(ssi_slope) + ' /mV')
        axs[0].text(-120, 0.6, 'Slope (Experimental): ' + str(ssi_slope_exp) + ' /mV')
        axs[0].text(-120, 0.5, 'V50 (Optimized): ' + str(v_half) + ' mV')
        axs[0].text(-120, 0.4, 'V50 (Experimental): ' + str(v_half_ssi_exp) + ' mV')
        axs[0].legend()

        # Activation curve
        gnorm_vec, v_vec, all_is = ggsdHMM.Activation(channel_name=self.channel_name, step=5).genActivation()
        gnorm_array = np.array(gnorm_vec)
        v_array = np.array(v_vec)
        gv_slope, v_half, top, bottom = cf.calc_act_obj(self.channel_name, is_HMM=True)

        even_xs = np.linspace(v_array[0], v_array[len(v_array)-1], 100)
        curve = cf.boltzmann(even_xs, gv_slope, v_half, top, bottom)
        axs[1].set_xlabel('Voltage (mV)')
        axs[1].set_ylabel('Fraction Activated')
        axs[1].set_title("Activation Curve")
        axs[1].scatter(v_array, gnorm_array, color='red',marker='s', label='Optimized Activation')
        axs[1].plot(even_xs, curve, color='red', label="Fitted Activation")
        curve_exp = cf.boltzmann(even_xs, gv_slope_exp, v_half_act_exp, top, bottom)
        #curve_exp = cf.boltzmann(even_xs, gv_slope_exp, v_half_act_exp, 1, 0)
        axs[1].plot(even_xs, curve_exp, color='black', label='Activation Experimental')
        axs[1].text(-120, 0.7, 'Slope (Optimized): ' + str(gv_slope) + ' /mV')
        axs[1].text(-120, 0.6, 'Slope (Experimental): ' + str(gv_slope_exp) + ' /mV')
        axs[1].text(-120, 0.5, 'V50 (Optimized): ' + str(v_half) + ' mV')
        axs[1].text(-120, 0.4, 'V50 (Experimental): ' + str(v_half_act_exp) + ' mV')
        axs[1].legend()
        
        # Recovery Curve
        rec_inact_tau_vec, recov_curves, times = ggsdHMM.RFI(channel_name=self.channel_name).genRecInactTau()
        times = np.array(times)
        data_pts = np.array(recov_curves[0])
        axs[2].set_xlabel('Log(Time)')
        axs[2].set_ylabel('Fractional Recovery')
        axs[2].set_title("Recovery from Inactivation")
        even_xs = np.linspace(times[0], times[len(times)-1], 100)
        y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(self.channel_name, is_HMM=True)
        curve = cf.two_phase(even_xs, y0, plateau, percent_fast, k_fast, k_slow)
        axs[2].plot(np.log(even_xs), curve, c='red',label="Recovery Fit")
        curve_exp = cf.two_phase(even_xs, y0, plateau, percent_fast_exp, 1/tau_fast_exp, 1/tau_slow_exp)
        axs[2].plot(np.log(even_xs), curve_exp, c='black')
        axs[2].scatter(np.log(times), data_pts, label='Optimized Recovery', color='black')
        
        axs[2].text(4, 0.9, 'Tau Fast (Optimized): ' + str(1/k_fast))
        axs[2].text(4, 0.85, 'Tau Fast (Experimental): ' + str(tau_fast_exp))
        axs[2].text(4, 0.8, 'Tau Slow (Optimized): ' + str(1/k_slow))
        axs[2].text(4, 0.75, 'Tau Slow (Experimental): ' + str(tau_slow_exp))
        axs[2].text(4, 0.7, 'Percent Fast (Optimized): ' + str(percent_fast))
        axs[2].text(4, 0.65, 'Percent Fast (Experimental): ' + str(percent_fast_exp))
        axs[2].legend()
        
        plt.show()

