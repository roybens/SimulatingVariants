import eval_helper as eh
import curve_fitting as cf
from scipy.stats import linregress
import math
import numpy as np
import eval_helper_na12mut as ehn
import eval_helper_na12mut8st as ehn8

class Score_Function:
    '''
    Class that calculates the score of any particular set of generated data
    compared to the wild data.

    Approach: For any individual parameter list from the population, extrapolate
    a number of points on each curve (activation, inactivation, and recovery). Do 
    the same for the wild type (same number of points). Calculate the sum squared 
    error between the two sets of points and return that as the error.
    '''
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
        self.tau0_wild = wild_data['tau0']
        # self.ramp_wild = wild_data['ramp']
        # self.persistent_wild = wild_data['persistent']
        # Hard-coded last-minute "objectives"
        self.peak_amp_wild = wild_data['peak_amp']
        
        self.channel_name = channel_name
        
    def total_rmse(self, is_HMM=False, objectives=['inact', 'act', 'recov']):
        # When using the HH model, leave is_HMM as false. Otherwise, set it to true.
        errors = []
        # Check to see if persistent, ramp, and udb20 are within reasonable ranges:
        # TODO UDB20 and ranges for each of the aforementioned values

        # ramp = ggsd.Ramp(channel_name=self.channel_name)
        # ramp_area = ramp.areaUnderCurve()
        # persistent_curr = ramp.persistentCurrent()

        # Baselines for ramp, persistent, and udb20

        # ramp_baseline = self.ramp_wild * self.ramp_diff / 100
        # persistent_baseline = self.persistent_wild * self.persistent_diff / 100
        # udb20_baseline = self.udb20_wild * self.udb20_diff / 100
        
        # Limits for these values are percentages of the baseline. Format: [lower %, upper %]
        # ramp_limits = [0, 100]  
        # persistent_limits = [0, 100]
        # udb20_limits = [0, 100]
        
        # ramp_diff = ramp_area / ramp_baseline * 100
        # persistent_diff = persistent_curr / persistent_baseline * 100
        # udb20_diff = udb20 / udb20_baseline * 100
        '''
        if not ramp_diff > ramp_limits[0] and ramp_diff < ramp_limits[1] \
                and persistent_diff > persistent_limits[0] and persistent_diff < persistent_limits[1] \
                and udb20_diff > udb20_limits[0] and udb20_diff < udb20_limits[1]:
            for i in range(len(objectives)):
                errors.append(1000)
            return errors
        '''
        if 'inact' in objectives:
            inact_err = self.calc_inact_err(is_HMM)
            errors.append(inact_err)
        if 'act' in objectives:
            act_err = self.calc_act_err(is_HMM)
            errors.append(act_err)
        if 'recov' in objectives:
            recov_err = self.calc_recov_err(is_HMM)
            errors.append(recov_err)
        if 'tau0' in objectives:
            tau0_err = self.calc_tau0_err(is_HMM)
            errors.append(tau0_err)
        if 'peak_amp' in objectives:
            peak_amp_err = self.calc_peak_amp_err(is_HMM)
            errors.append(peak_amp_err)
        return errors
            
    def calc_inact_err(self, is_HMM):
        try:
            ssi_slope, v_half_inact, top, bottom = cf.calc_inact_obj(self.channel_name, is_HMM)
        except ZeroDivisionError:
            print('Zero Division Error while calculating inact')
            return 1000
        except: 
            return 1000
        v_array = np.linspace(-120, 40, 20)
        #Calculate wild protocol values
        slope_wild = float(self.ssi_slope_wild)*float(self.ssi_slope_diff)/100
        v_half_wild = float(self.v_half_ssi_wild) + float(self.dv_half_ssi_diff)
        wild_curve = cf.boltzmann(v_array, slope_wild, v_half_wild, top, bottom)
        opt_curve = cf.boltzmann(v_array, ssi_slope, v_half_inact, top, bottom)
        
        error = sum([(wild_curve[i] - opt_curve[i])**2 for i in range(len(wild_curve))])
        return error
        
        
    def calc_act_err(self, is_HMM):
        try:
            gv_slope, v_half_act, top, bottom = cf.calc_act_obj(self.channel_name, is_HMM)
        except ZeroDivisionError:
            print('Zero Division Error while calculating act')
            return 1000
        except:
            return 1000
        v_array = np.linspace(-120, 40, 20)
        #Calculate wild protocol values
        slope_wild = float(self.gv_slope_wild)*float(self.gv_slope_diff)/100
        v_half_wild = float(self.v_half_act_wild) + float(self.dv_half_act_diff)
        wild_curve = cf.boltzmann(v_array, slope_wild, v_half_wild, top, bottom)
        opt_curve = cf.boltzmann(v_array, gv_slope, v_half_act, top, bottom)
        
        error = sum([(wild_curve[i] - opt_curve[i])**2 for i in range(len(wild_curve))])
        return error
        
    def calc_recov_err(self, is_HMM):
        try:
            y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(self.channel_name, is_HMM)
            if (y0, plateau, percent_fast, k_fast, k_slow) == (1000, 1000, 1000, 1000, 1000) or (y0, plateau, percent_fast, k_fast, k_slow) == (1, 1, 1, 1, 1) or y0 > 0.5:
                return 1000
        except ZeroDivisionError:
            print('Zero Division Error while calculating recov')
            return 1000
        except:
            return 1000
        t_array = np.linspace(1, 100, 50)
        # t_array = list(np.linspace(1, 100, 50))
        # t_array.append(list(np.linspace(101, 5000, 50)))
        # t_array = np.array(t_array)
        #Calculate wild protocol values
        tau_fast_wild = float(self.tau_fast_wild)*float(self.tau_fast_diff)/100
        tau_slow_wild = float(self.tau_slow_wild)*float(self.tau_slow_diff)/100
        percent_fast_wild = float(self.percent_fast_wild)*float(self.percent_fast_diff)/100
        wild_curve = cf.two_phase(t_array, 0, 1, percent_fast_wild, 1/tau_fast_wild, 1/tau_slow_wild)
        opt_curve = cf.two_phase(t_array, y0, plateau, percent_fast, k_fast, k_slow)
        errors = [(wild_curve[i] - opt_curve[i])**2 for i in range(len(wild_curve))]
        print(errors)
        error = sum(errors)
        if error == 0:
            print(y0)
            print(plateau)
            print(percent_fast)
            print(k_fast)
            print(k_slow)
        return error
    def calc_tau0_err(self, is_HMM):
        # tau0 = cf.calc_tau0_obj(self.channel_name, is_HMM)
        # tau0_wild = float(self.tau0_wild)*float(self.tau0_diff)/100
        # return (tau0 - tau0_wild)**2
        try:
            tau0 = cf.calc_tau0_obj(self.channel_name, is_HMM)
            tau0_wild = float(self.tau0_wild)*float(self.tau0_diff)/100
            return (tau0 - tau0_wild)**2
        except:
            print('Error when calculating tau0')
            return 1000
    
    def calc_peak_amp_err(self, is_HMM):
        try:
            peak_amp = cf.calc_peak_amp_obj(self.channel_name, is_HMM)
            peak_amp_wild = float(self.peak_amp_wild)
            print('peak_amp: ' + str(peak_amp))
            print('peak_amp_wild: ' + str(peak_amp_wild))
            return (peak_amp - peak_amp_wild)**2
        except:
            print('Error when calculating peak_amp')
            return 1000
