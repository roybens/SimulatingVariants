import numpy as np
import bluepyopt as bpop
import eval_helper as eh
import scoring_functions as sf

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

        exp_data_file: a filepath to a csv containing the experimental data, Israeli style. For 
                       other file formats like Northwestern, use a different READ_DATA function 
                       or write your own.
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
                #Try removing value from initialization
                param_list.append(bpop.parameters.Parameter(param_name, value=param_val, bounds=(min_bound, max_bound)))
                #param_list.append(bpop.parameters.Parameter(param_name, bounds=(min_bound, max_bound)))
                #Try setting all bounds to 0 and 1
                #param_list.append(bpop.parameters.Parameter(param_name, bounds=(0, 1)))


            return param_list

        self.params = init_params(params_file)
        self.objectives = [bpop.objectives.Objective('inact'),\
                           bpop.objectives.Objective('act'),\
                           bpop.objectives.Objective('recov'),
                           #bpop.objectives.Objective('tau0')
                           ]
        #exp_data_map = eh.read_all_raw_data(exp_data_file)
        exp_data_map = eh.read_all_raw_data_SCN8A(exp_data_file)
        self.target_data = exp_data_map[exp][mutant]
 

    def evaluate_with_lists(self, param_values=[]):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors
        '''
        return self.calc_all_rmse(param_values, sf.calc_rmse_sans_tau)
    

    def calc_all_rmse(self, param_values, scoring_function):
        '''
        Uses the parameter values in PARAM_VALUES to calculate the objective errors

        Arguments:
            param_values: list of float parameter values in order

        Returns:
            List of float values of objective errors

        '''
        assert len(param_values) == len(self.params), 'Parameter value list is not same length number of parameters' 
        eh.change_params(param_values, scaled=False)
        return scoring_function(self.target_data)
