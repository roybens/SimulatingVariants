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
    def __init__(self, protocols_dict):
        self.dv_half_act_protocol = protocols_dict['dv_half_act']
        self.gv_slope_protocol = protocols_dict['gv_slope']
        self.dv_half_ssi_protocol = protocols_dict['dv_half_ssi']
        self.ssi_slope_protocol = protocols_dict['ssi_slope']
        self.tau_fast_protocol = protocols_dict['tau_fast']
        self.tau_slow_protocol = protocols_dict['tau_slow']
        self.percent_fast_protocol = protocols_dict['percent_fast']
        self.udb20_protocol = protocols_dict['udb20']
        self.tau0_protocol = protocols_dict['tau0']
        self.ramp_protocol = protocols_dict['ramp']
        self.persistent_protocol = protocols_dict['persistent']


    def dv_half_act(self, plus_minus_wild):

    def gv_slope(self, percent_wild):

    def dv_half_ssi(self, plus_minus_wild):

    def ssi_slope(self, percent_wild):

    def tau_fast(self, percent_wild):

    def tau_slow(self, percent_wild):

    def percent_fast(self, percent_wild):

    def udb20(self, percent_wild):

    def tau0(self, percent_wild):

    def ramp(self, percent_wild):

    def persistent(self, percent_wild):


    def get_values_from_gensim():
        gen_data = eh.gen_sim_data()

        def find_half_and_slope(v_vec, ipeak_vec):
            """ Returns V50 and slope
                Notes:
                    gpeak_max = gpeak_vec.max() maximum value of the conductance used to normalize the conductance vector
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
            return v_half, s

        self.gv_slope, self.dv_half_act = find_half_and_slope(gen_data['act sweeps'], gen_data['act'])
        self.ssi_slope, self.dv_half_ssi = find_half_and_slope(gen_data['inact sweeps'], gen_data['inact'])
        self.tau_0 = gen_data['tau0']
        #self.udb20 ignore it for now
        #ramp and persistent not ready yet
