import numpy as np
import bluepyopt as bpop
import eval_helper as eh

class vclamp_evaluator(bpop.evaluators.Evaluator):
    '''
    A class that holds a set of objectives and a set of parameters.
    
    self.params holds the names of each of the parameters to be evaluated along with 
    their bounds and values

    self.objectives holds a set of categories for which an error will be calculated 
    through the evaluate_with_lists function
    '''

    def __init__(self, exp_data_file, params_file, exp, mutant):
        '''
        Constructor

        exp_data_file: a filepath to a csv containing the experimental data, NW style
        params_file: a filepath to a csv containing the names, starting values, and bounds
            of each parameter in the following format:

            parameter name | parameter value | lower bound | upper bound

                  sh       |         5       |      3      |      15
                  ...      |        ...      |     ...     |      ...

        exp: name of the experiment
        mutant: name of the mutant

        '''

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

        self.params = init_params(params_file)
        self.objectives = [bpop.objectives.Objective('inact'),\
                           bpop.objectives.Objective('act'),\
                           bpop.objectives.Objective('recov'),\
                           bpop.objectives.Objective('tau0')
                           ]
        exp_data_map = eh.read_all_raw_data(exp_data_file)
        self.target_data = exp_data_map[exp][mutant]
 

    def evaluate_with_lists(self, param_values=[]):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors
        '''
        return calc_all_rmse(param_values)
    

    def calc_all_rmse(param_values):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors

        '''
        assert len(param_values) == len(self.params), 'Parameter value list is not same length number of parameters' 
        eh.change_params(param_values)
        try:
            sim_data = eh.gen_sim_data()
        except ZeroDivisionError: #catch error to prevent bad individuals from halting run
            print("ZeroDivisionError when generating sim_data, returned infinity.")
            sim_data =None
            return (1000,1000,1000,1000)
        inds = self.target_data["inact sig inds"]
        squared_diffs = [(self.target_data[var][i]-sim_data[var][i])**2 for i in inds]
        inact_rmse = (sum(squared_diffs)/len(inds))**.5

        inds = self.target_data["act sig inds"]
        squared_diffs = [(self.target_data[var][i]-sim_data[var][i])**2 for i in inds]
        act_rmse = (sum(squared_diffs)/len(inds))**.5

        inds = self.target_data["recov sig inds"]
        squared_diffs = [(self.target_data[var][i]-sim_data[var][i])**2 for i in inds]
        recov_rmse = (sum(squared_diffs)/len(inds))**.5
        
        tau_rmse = ((self.target_data["tau0"]-sim_data["tau0"])**2)**.5

        return [inact_rmse, act_rmse, recov_rmse, tau_rmse]
