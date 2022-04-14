############################################
## Code for Hidden-Markov Model evaluator ##
########## Author: Michael Lam #############
############################################

import numpy as np
import bluepyopt as bpop
import eval_helper as eh
import scoring_functions_relative as sf  # Change this to scoring_functions_relative or scoring_functions_exp to change scoring functions
import curve_fitting as cf
import matplotlib.pyplot as plt
import generalized_genSim_shorten_time_HMM as ggsdHMM
import generalized_genSim_shorten_time as ggsd
import eval_helper_na12mut as ehn
import eval_helper_na12mut8st as ehn8


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
        act_obj = ggsd.Activation(channel_name=self.channel_name_HH)
        tau0 = ehn.find_tau0(act_obj)
        peak_amp = ehn.find_peak_amp(act_obj)
        time_to_peak = ehn.find_time_to_peak(act_obj)
        act_obj = ggsd.Activation(channel_name=self.channel_name_HH)
        
        #recov_obj = ggsdHMM.RFI(channel_name=self.channel_name_HH)
        gv_slope, v_half_act, top, bottom = cf.calc_act_obj(act_obj)
        inact_obj = ggsd.Inactivation(channel_name=self.channel_name_HH)
        ssi_slope, v_half_inact, top, bottom = cf.calc_inact_obj(inact_obj)
        #y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(recov_obj)
        # gv_slope, v_half_act, top, bottom = (1, 1, 1, 1)
        # ssi_slope, v_half_inact, top, bottom = (1, 1, 1, 1)
        # y0, plateau, percent_fast, k_fast, k_slow = (1, 1, 1, 1, 1)
        tau0 = ehn.find_tau0(act_obj)
        peak_amp = ehn.find_peak_amp(act_obj)
        time_to_peak = ehn.find_time_to_peak(act_obj)
        # Ramp Protocol
        # ramp = ggsdHMM.Ramp(channel_name=self.channel_name)
        # ramp_area = ramp.areaUnderCurve
        # persistent_curr = ramp.persistentCurrent()

        wild_data['v_half_act'] = v_half_act
        wild_data['gv_slope'] = gv_slope
        wild_data['v_half_ssi'] = v_half_inact
        wild_data['ssi_slope'] = ssi_slope
        #wild_data['tau_fast'] = 1 / k_fast
        #wild_data['tau_slow'] = 1 / k_slow
        #wild_data['percent_fast'] = percent_fast
        # wild_data['udb20'] = 0
        wild_data['tau0'] = tau0
        # wild_data['ramp'] = ramp_area
        # wild_data['persistent'] = persistent_curr
        
        # Some extra objectives added last minute, so this is a bit hard-coded
        wild_data['peak_amp'] = peak_amp
        wild_data['time_to_peak'] = time_to_peak
        self.wild_data = wild_data
        print(wild_data)
        return wild_data


    def evaluate_with_lists(self, param_values=[]):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors
        '''
        errors = []
        act_obj = ggsdHMM.Activation(channel_name=self.channel_name_HMM)
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=act_obj)
        gv_slope, v_half_act, top, bottom = cf.calc_act_obj(act_obj)

        inact_obj = ggsdHMM.Inactivation(channel_name=self.channel_name_HMM)
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=inact_obj)
        ssi_slope, v_half_inact, top, bottom = cf.calc_inact_obj(inact_obj)
        #'v_half_act', 'gv_slope', 'v_half_ssi', 'ssi_slope', 'tau0', 'ttp', 'peak_current']
        if 'v_half_act' in self.objective_names:
            vhalf_act_error = (v_half_act - self.wild_data['v_half_act'])**2
            errors.append(vhalf_act_error)
        if 'gv_slope' in self.objective_names:
            gv_slope_error = (gv_slope - self.wild_data['gv_slope'])**2
            errors.append(gv_slope_error)
        if 'v_half_ssi' in self.objective_names:
            v_half_ssi_error = (v_half_inact - self.wild_data['v_half_ssi'])**2
            errors.append(v_half_ssi_error)
        if 'ssi_slope' in self.objectives:
            ssi_slope_error = (ssi_slope - self.wild_data['ssi_slope'])**2
            errors.append(ssi_slope_error)
        if 'tau0' in self.objective_names:
            act_obj = ggsdHMM.Activation(channel_name=self.channel_name_HMM)
            eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=inact_obj)
            try:
                tau0 = ehn.find_tau0(act_obj)
                tau0_error = (tau0 - self.wild_data['tau0'])**2
            except:
                print('tau got 1000')
                tau0_error = 1000
            errors.append(tau0_error*100)
        if 'peak_current' in self.objective_names:
            peak_amp = ehn.find_peak_amp(act_obj)
            peak_amp_errors = np.sum([(peak_amp[i] - self.wild_data['peak_amp'][i])**2 for i in range(len(peak_amp))])
            errors.append(peak_amp_error)
        if 'ttp' in self.objective_names:
            time_to_peak = ehn.find_time_to_peak(act_obj)
            time_to_peak_error = np.sum([(time_to_peak[i] - self.wild_data['time_to_peak'][i])**2 for i in range(len(peak_amp))])
            errors.append(time_to_peak_error*100)
        print(errors)
        print(self.objectives)
        #python3 Optimization_HHtoHMM_rel.pypython3 Optimization_HHtoHMM_rel.py
        return errors
    

    def calc_all_rmse(self, param_values):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors

        '''
        assert len(param_values) == len(self.params), 'Parameter value list is not same length number of parameters' 
        act_obj = ggsdHMM.Activation(channel_name=self.channel_name_HMM)
        inact_obj = ggsdHMM.Inactivation(channel_name=self.channel_name_HMM)
        # recov_obj = ggsdHMM.RFI(channel_name=self.channel_name_HMM)
        recov_obj = None
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=act_obj)
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=inact_obj)
        score = self.score_calculator.total_rmse(act_obj, inact_obj, None, is_HMM=True, objectives=self.objective_names)
        # print((param_values, score))
        return score
    
    def plot_inact(self, param_values):
        fig, axs = plt.subplots(1, figsize=(10,10))
        inact_obj = ggsdHMM.Inactivation(channel_name=self.channel_name_HMM)
        # Inactivation curve
        # Calculate wild baseline values
        v_half_ssi_exp = self.wild_data['v_half_ssi'] + float(self.protocols['dv_half_ssi'])
        ssi_slope_exp = self.wild_data['ssi_slope'] * float(self.protocols['ssi_slope']) / 100
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=inact_obj)
        inorm_vec, v_vec, all_is = inact_obj.genInactivation()
        inorm_array = np.array(inorm_vec)
        v_array = np.array(v_vec)

        ssi_slope, v_half, top, bottom = cf.calc_inact_obj(inact_obj)

        
        even_xs = np.linspace(v_array[0], v_array[len(v_array)-1], 100)
        curve = cf.boltzmann(even_xs, ssi_slope, v_half, top, bottom)
        axs.set_xlabel('Voltage (mV)')
        axs.set_ylabel('Fraction Inactivated')
        axs.set_title("Inactivation Curve")
        axs.scatter(v_array, inorm_array, color='red', marker='s', label='Optimized Inactivation')
        axs.plot(even_xs, curve, color='red', label="Fitted Inactivation")

        
        curve_exp = cf.boltzmann(even_xs, ssi_slope_exp, v_half_ssi_exp, 0, 1)
        axs.plot(even_xs, curve_exp, color='black', label='Inactivation experimental')
        axs.text(-.120, 0.7, 'Slope (Optimized): ' + str(ssi_slope) + ' /mV')
        axs.text(-.120, 0.6, 'Slope (Experimental): ' + str(ssi_slope_exp) + ' /mV')
        axs.text(-.120, 0.5, 'V50 (Optimized): ' + str(v_half) + ' mV')
        axs.text(-.120, 0.4, 'V50 (Experimental): ' + str(v_half_ssi_exp) + ' mV')
        axs.legend()
        plt.show()

    
    def plot_act(self, param_values):
        act_obj = ggsdHMM.Activation(channel_name=self.channel_name_HMM)
        fig, axs = plt.subplots(1, figsize=(10,10))
        # Calculate wild baseline values
        v_half_act_exp = self.wild_data['v_half_act'] + float(self.protocols['dv_half_act'])
        gv_slope_exp = self.wild_data['gv_slope'] + float(self.protocols['gv_slope']) / 100
        
        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=act_obj)     
        gnorm_vec, v_vec, all_is = act_obj.genActivation()
        gnorm_array = np.array(gnorm_vec)
        v_array = np.array(v_vec)
        gv_slope, v_half, top, bottom = cf.calc_act_obj(act_obj)

        even_xs = np.linspace(v_array[0], v_array[len(v_array)-1], 100)
        curve = cf.boltzmann(even_xs, gv_slope, v_half, top, bottom)
        axs.set_xlabel('Voltage (mV)')
        axs.set_ylabel('Fraction Activated')
        axs.set_title("Activation Curve")
        axs.scatter(v_array, gnorm_array, color='red',marker='s', label='Optimized Activation')
        axs.plot(even_xs, curve, color='red', label="Fitted Activation")
        
        
        curve_exp = cf.boltzmann(even_xs, gv_slope_exp, v_half_act_exp, 1, 0)
        axs.plot(even_xs, curve_exp, color='black', label='Activation Experimental')
        axs.text(-.120, 0.7, 'Slope (Optimized): ' + str(gv_slope) + ' /mV')
        axs.text(-.120, 0.6, 'Slope (Experimental): ' + str(gv_slope_exp) + ' /mV')
        axs.text(-.120, 0.5, 'V50 (Optimized): ' + str(v_half) + ' mV')
        axs.text(-.120, 0.4, 'V50 (Experimental): ' + str(v_half_act_exp) + ' mV')
        axs.legend()
        plt.show()
        
    def plot_rec(self, param_values):
        fig, axs = plt.subplots(1, figsize=(10,10))
        tau_fast_exp = self.wild_data['tau_fast'] * float(self.protocols['tau_fast']) / 100
        tau_slow_exp = self.wild_data['tau_slow']  * float(self.protocols['tau_slow']) / 100
        percent_fast_exp = self.wild_data['percent_fast'] * float(self.protocols['percent_fast']) / 100

        eh.change_params(param_values, scaled=False, is_HMM=True, sim_obj=self.act_obj)
        rec_inact_tau_vec, recov_curves, times = self.recov_obj.genRecInactTau()
        times = np.array(times)
        data_pts = np.array(recov_curves[0])
        even_xs = np.linspace(times[0], times[len(times)-1], 10000)
        y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(self.channel_name_HMM, self.recov_obj)
        curve = cf.two_phase(even_xs, y0, plateau, percent_fast, k_fast, k_slow)
        # red curve: using the optimization fitted params to plot on the given x values
        axs.plot(np.log(even_xs), curve, c='red',label="Recovery Fit")
        # curve_exp = cf.two_phase(even_xs, y0, plateau, percent_fast_exp, 1/tau_fast_exp, 1/tau_slow_exp)
        curve_exp = cf.two_phase(even_xs, 0, 1, percent_fast_exp, 1/tau_fast_exp, 1/tau_slow_exp)
        print(y0)
        print(plateau)
        print(percent_fast_exp)
        print(tau_fast_exp)
        print(tau_slow_exp)
        # black curve: plotted from given data
        axs.plot(np.log(even_xs), curve_exp, c='black')
        # red dots: plot the given data points
        print(np.log(times))
        print(data_pts)
        axs.scatter(np.log(times), data_pts, label='Optimized Recovery', color='red')
        
        axs.set_xlabel('Log(Time)')
        axs.set_ylabel('Fractional Recovery')
        axs.set_title("Recovery from Inactivation")
        axs.text(4, 0.9, 'Tau Fast (Optimized): ' + str(1/k_fast))
        axs.text(4, 0.85, 'Tau Fast (Experimental): ' + str(tau_fast_exp))
        axs.text(4, 0.8, 'Tau Slow (Optimized): ' + str(1/k_slow))
        axs.text(4, 0.75, 'Tau Slow (Experimental): ' + str(tau_slow_exp))
        axs.text(4, 0.7, 'Percent Fast (Optimized): ' + str(percent_fast))
        axs.text(4, 0.65, 'Percent Fast (Experimental): ' + str(percent_fast_exp))
        axs.legend()
        
    def unit_test(self):
        inorm_vec, v_vec, all_is = ggsdHMM.Inactivation(channel_name=self.channel_name, step=5).genInactivation()
        plot_inact(self)
        plot_act(self)


# if __name__ == '__main__':
    # vce = Vclamp_evaluator_HMM('./param_stats_narrow.csv', 'A427D', 'na', objective_names=['inact', 'act', 'recov'])
    # print(vce.channel_name)
    # vce.unit_test()
    
