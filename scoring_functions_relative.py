import eval_helper as eh
import mutant_protocols as mp
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


}





