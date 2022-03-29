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
    def __init__(self, diff_dict, wild_data, channel_name):
        # Initiation of the scoring function is the same regardless of whether 
        # we're using an HMM or HH model.
        self.dv_half_act_diff = diff_dict['dv_half_act']
        self.gv_slope_diff = diff_dict['gv_slope']
        self.dv_half_ssi_diff = diff_dict['dv_half_ssi']
        self.ssi_slope_diff = diff_dict['ssi_slope']
        self.tau_fast_diff = diff_dict['tau_fast']
        self.tau_slow_diff = diff_dict['tau_slow']
        self.percent_fast_diff = diff_dict['percent_fast']
        # self.udb20_diff = diff_dict['udb20']
        self.tau0_diff = diff_dict['tau0']
        # self.ramp_diff = diff_dict['ramp']
        # self.persistent_diff = diff_dict['persistent']

        self.v_half_act_wild = wild_data['v_half_act']
        self.gv_slope_wild = wild_data['gv_slope']
        self.v_half_ssi_wild = wild_data['v_half_ssi']
        self.ssi_slope_wild = wild_data['ssi_slope']
        self.tau_fast_wild = wild_data['tau_fast']
        self.tau_slow_wild = wild_data['tau_slow']
        self.percent_fast_wild = wild_data['percent_fast']
        # self.udb20_wild = wild_data['udb20']
        # self.tau0_wild = wild_data['tau0']
        # self.ramp_wild = wild_data['ramp']
        # self.persistent_wild = wild_data['persistent']
        
        self.channel_name = channel_name


    def total_rmse(self, act_obj, inact_obj, recov_obj, is_HMM=False, objectives=['v_half_act', 'gv_slope', 'v_half_ssi', 'ssi_slope', 'tau_fast', 'tau_slow', 'percent_fast', 'udb20', 'tau0', 'ramp', 'persistent']):
        # When using the HH model, leave is_HMM as false. Otherwise, set it to true.
        try:
            gv_slope, v_half_act, top, bottom = cf.calc_act_obj(act_obj)
            ssi_slope, v_half_inact, top, bottom = cf.calc_inact_obj(inact_obj)
            # y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(recov_obj)
                
        except ZeroDivisionError:
            print('Zero Division Error')
            error_val = []
            for i in range(len(objectives)):
                error_val.append(1000)
            return tuple(error_val) 
        errors = []
        if 'v_half_act' in objectives:
            v_half_act_err = self.dv_half_act(self.dv_half_act_diff, v_half_act)
            errors.append(v_half_act_err)
        if 'gv_slope' in objectives:
            gv_slope_err = self.gv_slope(self.gv_slope_diff, gv_slope)
            errors.append(gv_slope_err)
        if 'v_half_ssi' in objectives:
            v_half_ssi_err = self.dv_half_ssi(self.dv_half_ssi_diff, v_half_inact)
            errors.append(v_half_ssi_err)
        if 'ssi_slope' in objectives:
            ssi_slope_err = self.ssi_slope(self.ssi_slope_diff, ssi_slope)
            errors.append(ssi_slope_err)
        if 'tau_fast' in objectives:
            tau_fast_err = self.tau_fast(self.tau_fast_diff, 1/k_fast)
            errors.append(tau_fast_err)
        if 'tau_slow' in objectives:
            tau_slow_err = self.tau_slow(self.tau_slow_diff, 1/k_slow)
            errors.append(tau_slow_err)
        if 'percent_fast' in objectives:
            percent_fast_err = self.percent_fast(self.percent_fast_diff, percent_fast)
            errors.append(percent_fast_err)
        if 'udb20' in objectives:
            udb20_err = 0
            errors.append(udb20_err)
        if 'tau0' in objectives:
            tau0_err = self.tau0(self.tau0_diff, tau0)
            errors.append(tau0_err)
        if 'ramp' in objectives:
            ramp_err = 0
            errors.append(ramp_err)
        if 'persistent' in objectives:
            persistent_err = 0
            errors.append(persistent_err)

        return tuple(errors)
        
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
