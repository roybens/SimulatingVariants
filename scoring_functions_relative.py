import eval_helper as eh
import mutant_protocols as mp
from scipy.stats import linregress
'''
This file contains various scoring functions that can be used in the evaluator's
calc_rmse function. These functions tend follow the "relative" format where one 
specifies the how to score a value "relative" to its different from the wild type
according to some mod file specifications. At the time of writing, these models 
are being specified for na12_mut.mod.
'''

class Score_Function:
    def __init__(self, diff_dict, wild_data):
        self.dv_half_act_diff = diff_dict['dv_half_act']
        self.gv_slope_diff = diff_dict['gv_slope']
        self.dv_half_ssi_diff = diff_dict['dv_half_ssi']
        self.ssi_slope_diff = diff_dict['ssi_slope']
        self.tau_fast_diff = diff_dict['tau_fast']
        self.tau_slow_diff = diff_dict['tau_slow']
        self.percent_fast_diff = diff_dict['percent_fast']
        self.udb20_diff = diff_dict['udb20']
        self.tau0_diff = diff_dict['tau0']
        self.ramp_diff = diff_dict['ramp']
        self.persistent_diff = diff_dict['persistent']

        self.v_half_act_wild = wild_data['v_half_act']
        self.gv_slope_wild = wild_data['gv_slope']
        self.v_half_ssi_wild = wild_data['v_half_ssi']
        self.ssi_slope_wild = wild_data['ssi_slope']
        self.tau_fast_wild = wild_data['tau_fast']
        self.tau_slow_wild = wild_data['tau_slow']
        self.percent_fast_wild = wild_data['percent_fast']
        self.udb20_wild = wild_data['udb20']
        self.tau0_wild = wild_data['tau0']
        self.ramp_wild = wild_data['ramp']
        self.persistent_wild = wild_data['persistent']


    def total_error(self):
        


    def dv_half_act(self, plus_minus_wild, v_half):
        #TODO Normalize for all of these
        v_half_baseline = self.v_half_act_wild + plus_minus_wild
        return (v_half - v_half_baseline)**2

    def gv_slope(self, percent_wild, gv_slope):
        gv_slope_baseline = self.gv_slope_wild * percent_wild
        return (gv_slope - gv_slope_baseline)**2

    def dv_half_ssi(self, plus_minus_wild):

    def ssi_slope(self, percent_wild):

    def tau_fast(self, percent_wild):

    def tau_slow(self, percent_wild):

    def percent_fast(self, percent_wild):

    #def udb20(self, percent_wild):

    def tau0(self, percent_wild):

    #def ramp(self, percent_wild):

    #def persistent(self, percent_wild):


    def get_values_from_gensim(k_fast, k_slow, span_fast, span_slow):
        '''
        Calculates various values for the scoring function, and stores the result as member variables of the object.
        ---
        Args:
            k_fast and k_slow are the two rate constant, expressed in reciprocal of the X axis time units. If X is in minutes, then K is expressed in inverse minute
        '''
        gen_data = eh.gen_sim_data()

        def find_half_and_slope(v_vec, ipeak_vec):
            """ 
            Returns V50 and slope
            """
            # convert to numpy arrays
            v_vec = np.array(v_vec)
            ipeak_vec = np.array(ipeak_vec)

            # find start of linear portion (0 mV and onwards)
            inds = np.where(v_vec >= 0)

            # take linear portion of voltage and current relationship
            lin_v = v_vec[inds]
            lin_i = ipeak_vec[inds]
            
            #boltzmann for conductance
            def boltzmann(vm, Gmax, v_half, s):
                vrev = stats.linregress(lin_i, lin_v).intercept
                return Gmax * (vm - vrev) / (1 + np.exp((v_half - vm) / s))

            Gmax, v_half, s = optimize.curve_fit(boltzmann, v_vec, ipeak_vec)[0]
            return s, v_half

        self.gv_slope, self.dv_half_act = find_half_and_slope(gen_data['act sweeps'], gen_data['act'])
        self.ssi_slope, self.dv_half_ssi = find_half_and_slope(gen_data['inact sweeps'], gen_data['inact'])
        self.tau_0 = gen_data['tau0']
        self.tau_fast = 1 / k_fast
        self.tau_slow = 1 / k_slow
        self.percent_fast = span_fast / span_slow
        #self.udb20: ignore it for now
        #ramp and persistent: not ready yet
