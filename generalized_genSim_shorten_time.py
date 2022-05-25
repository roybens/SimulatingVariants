"""
Written by Emily Nguyen, UC Berkeley
           Chastin Chung, UC Berkeley
           Isabella Boyle, UC Berkeley
           Roy Ben-Shalom, UCSF
    
Generates simulated data.
Modified from Emilio Andreozzi "Phenomenological models of NaV1.5.
    A side by side, procedural, hands-on comparison between Hodgkin-Huxley and kinetic formalisms." 2019
"""
import scipy
from neuron import h, gui
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import optimize, stats
from scipy.signal import find_peaks
import argparse
import os
import pickle
from generate_simulation import *

import optimize_na_ga_v2 as opt
import curve_fitting as cf
import eval_helper as eh

# from sys import api_version
# from test.pythoninfo import collect_platform


##################
# Global
##################
# Create folder in CWD to save plots
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'Plots_Folder')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)


##################
# Activation
##################
class Activation(Activation_general):
        
    def clamp(self, v_cl):
        """ Runs a trace and calculates peak currents.
        Args:
            v_cl (int): voltage to run
        """
        curr_tr = 0  # initialization of peak current
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.
        pre_i = 0  # initialization of variables used to commute the peak current
        dens = 0
        self.f3cl.amp[1] = v_cl  # mV

        for _ in self.ntrials:
            while h.t < h.tstop:  # runs a single trace, calculates peak current
                dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                    0.5).i_cap  # clamping current in mA/cm2, for each dt

                self.t_vec.append(h.t)
                self.v_vec_t.append(self.soma.v)
                self.i_vec.append(dens)

                if (h.t > 5) and (h.t <= 10):  # evaluate the peak
                    if abs(dens) > abs(pre_i):
                        curr_tr = dens  # updates the peak current

                h.fadvance()
                pre_i = dens

                # find i peak of trace
        peak,ttp = self.find_ipeaks()
        self.ipeak_vec.append(peak)
        self.ttp_vec.append(ttp)




##################
# Inactivation
##################
class Inactivation(Inactivation_general):

    def clamp(self, v_cl):
        """ Runs a trace and calculates peak currents.
        Args:
            v_cl (int): voltage to run
        """
        self.f3cl.amp[1] = v_cl
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.

        # parameters initialization
        peak_curr = 0
        dtsave = h.dt

        for _ in self.ntrials:
            while h.t < h.tstop:  # runs a single trace, calculates peak current
                if (h.t > 537) or (h.t < 40):
                    h.dt = dtsave
                else:
                    h.dt = 1
                self.update_clamp_time_step()

                h.fadvance()

        # find i peak of trace
        self.ipeak_vec.append(self.find_ipeaks())
        
        
    def clamp_at_voltage(self, v_cl):
        """ Runs a trace and calculates peak currents.
        Args:
            v_cl (int): voltage to run
        """
        self.f3cl.amp[1] = v_cl
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.

        # parameters initialization
        peak_curr = 0
        dtsave = h.dt

        for _ in self.ntrials:
            while h.t < h.tstop:  # runs a single trace, calculates peak current
                if (h.t > 537) or (h.t < 40):
                    h.dt = dtsave
                else:
                    h.dt = 1
                self.update_clamp_time_step()

                h.fadvance()

        # find i peak of trace
        self.ipeak_vec.append(self.find_ipeaks())

    def find_ipeaks(self):
        """
        Evaluate the peak and updates the peak current.
        Returns peak current.
        """
        return self.find_ipeaks_with_index()[1]
    
    def find_ipeaks_with_index(self):
        """
        Evaluate the peak and updates the peak current.
        Returns peak current.
        """
        # find peaks
        self.i_vec = np.array(self.i_vec)
        self.t_vec = np.array(self.t_vec)
        mask = np.where(np.logical_and(self.t_vec >= 535, self.t_vec <= 545))  # h.t window to take peak
        i_slice = self.i_vec[mask]
        peak_indices, properties_dict = find_peaks(i_slice * -1, height=0.1)  # find minima
        if len(peak_indices) == 0:
            peak_curr = 0
            return (-1, peak_curr)
        else:
            peak_curr = i_slice[peak_indices][0]
            return peak_indices[0], peak_curr

    def one_phase(self, x, y0, plateau, k):
        '''
        Fit a one-phase association curve to an array of data points X. 
        For info about the parameters, visit 
        https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_exponential_association.htm    
        '''
        return y0 + (plateau - y0) * (1 - np.exp(-k * x))

##################
# Recovery from Inactivation (RFI)
# &  RFI Tau
##################
class RFI(RFI_general):
    
    def find_ipeaks(self, start_ht, end_ht):
        """
        Evaluate the peak and updates the peak current.
        Returns peak current.
        Args:
            start_ht (int): h.t time of window to find peak
            end_ht (int): h.t time to end window to find peak
        """
        # find peaks
        self.i_vec_t = np.array(self.i_vec_t)
        self.t_vec = np.array(self.t_vec)
        mask = np.where(np.logical_and(self.t_vec >= start_ht, self.t_vec <= end_ht))  # h.t window to take peak
        i_slice = self.i_vec_t[mask]
        peak_indices, properties_dict = find_peaks(i_slice * -1, height=0.1)  # find minima
        if len(peak_indices) == 0:
            peak_curr = 0
        else:
            peak_curr = i_slice[peak_indices][0]
        return peak_curr

   


##################
# Ramp Protocol
##################
class Ramp(Ramp_general):
    

    def persistentCurrent(self):
        """ Calculates persistent current (avg current of last 100 ms at 0 mV)
        Normalized by peak from IV (same number as areaUnderCurve).
        """
        cutoff_start = self.t_end_persist
        cutoff_end = len(self.t_vec) - 1
        # remove current spike at end of 0 mV
        while np.abs(self.i_vec[cutoff_start + 1] - self.i_vec[cutoff_start]) < 1E-5:
            cutoff_start += 1
            if cutoff_start == cutoff_end:
                break
        while np.abs(self.i_vec[cutoff_end] - self.i_vec[cutoff_end - 1]) < 1E-5:
            cutoff_end -= 1
            if cutoff_end == self.t_end_persist:
                break

        persistent = self.i_vec[self.t_start_persist:self.t_end_persist]

        #create time vector for graphing current
        t_current = np.concatenate((self.t_vec[1:cutoff_start], self.t_vec[cutoff_end:]))
        self.i_vec = np.concatenate((self.i_vec[1:cutoff_start], self.i_vec[cutoff_end:]))

        act = Activation()
        act.genActivation()
        IVPeak = min(act.ipeak_vec)

        return ((sum(persistent) / len(persistent)) / IVPeak), t_current

 

##################
# UDB20 Protocol
##################
class UDB20(UDB20_general):
    def placeholder(self):
        return None


##################
# RFI dv tau
##################
class RFI_dv(RFI_dv_general):
    def placeholder(self):
        return None

