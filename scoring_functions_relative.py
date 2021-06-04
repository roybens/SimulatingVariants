import eval_helper as eh
#import mutant_protocols as mp
import curve_fitting as cf
from scipy.stats import linregress
import math
'''
This file contains various scoring functions that can be used in the evaluator's
calc_rmse function. These functions tend follow the "relative" format where one 
specifies the how to score a value "relative" to its different from the wild type
according to some mod file specifications. At the time of writing, these models 
are being specified for na12_mut.mod.
'''

class Score_Function:
    def __init__(self, diff_dict, wild_data):
        # Initiation of the scoring function is the same regardless of whether 
        # we're using an HMM or HH model.
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


    def total_rmse(self, is_HMM=False):
        # When using the HH model, leave is_HMM as false. Otherwise, set it to true.
        try:
            gv_slope, v_half_act, top, bottom = cf.calc_act_obj(is_HMM=is_HMM)
            print('gv_slope: ' + str(gv_slope))
            print('v_half_act: ' + str(v_half_act))
            ssi_slope, v_half_inact, top, bottom, tau0 = cf.calc_inact_obj(is_HMM=is_HMM)
            y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(is_HMM=is_HMM)
        except ZeroDivisionError:
            print('Zero Division Error')
            return (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)
 
        v_half_act_err = self.dv_half_act(self.dv_half_act_diff, v_half_act)

        gv_slope_err = self.gv_slope(self.gv_slope_diff, gv_slope)

        v_half_ssi_err = self.dv_half_ssi(self.dv_half_ssi_diff, v_half_inact)

        ssi_slope_err = self.ssi_slope(self.ssi_slope_diff, ssi_slope)

        tau_fast_err = self.tau_fast(self.tau_fast_diff, 1/k_fast)

        tau_slow_err = self.tau_slow(self.tau_slow_diff, 1/k_slow)

        percent_fast_err = self.percent_fast(self.percent_fast_diff, percent_fast)

        udb20_err = 0

        tau0_err = self.tau0(self.tau0_diff, tau0)

        ramp_err = 0
        
        persistent_err = 0
        #return (v_half_act_err, gv_slope_err, v_half_ssi_err, ssi_slope_err, tau_fast_err, tau_slow_err, percent_fast_err, udb20_err, tau0_err, ramp_err, persistent_err)
        return (v_half_act_err, gv_slope_err, v_half_ssi_err, ssi_slope_err, tau_fast_err, tau_slow_err)
        #return (tau0,)

    def dv_half_act(self, plus_minus_wild, v_half):
        try:
            v_half_baseline = float(self.v_half_act_wild) + float(plus_minus_wild)
            result = ((float(v_half) - v_half_baseline)/v_half_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except: 
            print('v_half_act_error')
            return 1000

    def gv_slope(self, percent_wild, gv_slope):
        try:
            gv_slope_baseline = float(self.gv_slope_wild) * float(percent_wild) / 100
            result = ((float(gv_slope) - gv_slope_baseline)/gv_slope_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('gv_slope_error')
            return 1000

    def dv_half_ssi(self, plus_minus_wild, v_half_ssi):
        try:
            v_half_baseline = float(self.v_half_ssi_wild) + float(plus_minus_wild)
            result = ((float(v_half_ssi) - v_half_baseline)/v_half_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('v_half_ssi_error')
            return 1000

    def ssi_slope(self, percent_wild, ssi_slope_exp):
        try:
            ssi_slope_baseline = float(self.ssi_slope_wild) * float(percent_wild) / 100
            result = ((float(ssi_slope_exp) - ssi_slope_baseline)/ssi_slope_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('ssi_slope_error')
            return 1000

    def tau_fast(self, percent_wild, tau_fast_exp):
        try:
            tau_fast_baseline = float(self.tau_fast_wild)*float(percent_wild) / 100
            result = ((float(tau_fast_exp) - tau_fast_baseline)/tau_fast_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('tau_fast_error')
            return 1000

    def tau_slow(self, percent_wild, tau_slow_exp):
        try:
            tau_slow_baseline = float(self.tau_slow_wild)*float(percent_wild) / 100
            result = ((float(tau_slow_exp) - tau_slow_baseline)/tau_slow_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('tau_slow_error')
            return 1000

    def percent_fast(self, percent_wild, percent_fast_exp):
        try:
            percent_fast_baseline = float(self.percent_fast_wild)*float(percent_wild) / 100
            result = ((float(percent_fast_exp) - percent_fast_baseline)/percent_fast_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('percent_fast_error')
            return 1000

    #def udb20(self, percent_wild):

    def tau0(self, percent_wild, tau0_exp):
        try:
            tau0_baseline = float(self.tau0_wild)*float(percent_wild) / 100
            result = ((float(tau0_exp) - tau0_baseline)/tau0_baseline)**2
            if math.isnan(result):
                return 1000
            return result
        except:
            print('tau0_error')
            return 1000


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
