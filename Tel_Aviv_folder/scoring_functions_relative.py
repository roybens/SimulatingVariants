import eval_helper as eh
import curve_fitting_tel_aviv as cf
import yaml
from scipy.stats import linregress
import math
'''
This file contains various scoring functions that can be used in the evaluator's
calc_rmse function. These functions tend follow the "relative" format where one 
specifies the how to score a value "relative" to its different from the wild type
according to some mod file specifications. At the time of writing, these models 
are being specified for na12_mut.mod.

Authors: Michael Lam
         Chastin Chung
'''

class Score_Function:
    def __init__(self, diff_dict, wild_data):
        self.dv_half_act_diff = diff_dict['dv_half_act']
        self.gv_slope_diff = diff_dict['gv_slope']
        self.dv_half_ssi_diff = diff_dict['dv_half_ssi']
        self.ssi_slope_diff = diff_dict['ssi_slope']
        self.tau_diff = diff_dict['tau']
        #self.persistent_diff = diff_dict['persistent10']
        #self.persistent_diff = diff_dict['persistent20']

        self.v_half_act_wild = wild_data['v_half_act']
        self.gv_slope_wild = wild_data['gv_slope']
        self.v_half_ssi_wild = wild_data['v_half_ssi']
        self.ssi_slope_wild = wild_data['ssi_slope']
        self.tau_wild = wild_data['tau']
        #self.persistent_wild = wild_data['persistent10']
        #self.persistent_wild = wild_data['persistent20']


    def total_rmse(self, tel_aviv_data):
        try:
            gv_slope, v_half_act, top, bottom = cf.calc_act_obj(channel_name='na8xst')
            ssi_slope, v_half_inact, top, bottom = cf.calc_inact_obj(channel_name='na8xst')
            y0, plateau, k, tau = cf.calc_recov_obj(channel_name='na8xst')
        except ZeroDivisionError:
            print('Zero Division Error')
            return (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)

        with open(tel_aviv_data, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        v_half_act_err = data[2][1]
        v_half_ssi_err = data[3][1]
        tau_err = data[4][1]
        gv_slope_err = data[5][1]
        ssi_slope_err = data[6][1]
        persistent10_err = data[7][1]
        persistent20_err = data[8][1]
        return (v_half_act_err, gv_slope_err, v_half_ssi_err, ssi_slope_err, tau_err)




    def get_values_from_gensim(k, span_fast, span_slow):
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
        self.tau = 1/k
        #self.udb20: ignore it for now
        #ramp and persistent: not ready yet
