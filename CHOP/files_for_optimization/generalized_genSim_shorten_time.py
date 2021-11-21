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
import generalized_genSim_shorten_time as ggsd

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
class Activation:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na16', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.025, ntrials=range(30),
                 dur=20, step=5, st_cl=-120, end_cl=40, v_cl=-120,
                 f3cl_dur0=5, f3cl_amp0=-120, f3cl_dur2=5, f3cl_amp2=-120,
                 ):

        self.h = h  # NEURON h

        # one-compartment cell (soma)
        self.channel_name = channel_name
        self.soma = h.Section(name='soma2')
        self.soma.diam = soma_diam  # micron
        self.soma.L = soma_L  # micron, so that area = 10000 micron2
        self.soma.nseg = soma_nseg  # adimensional
        self.soma.cm = soma_cm  # uF/cm2
        self.soma.Ra = soma_Ra  # ohm-cm
        self.soma.insert(channel_name)  # insert mechanism
        self.soma.ena = soma_ena

        # clamping parameters
        self.ntrials = ntrials  #
        h.celsius = h_celsius  # temperature in celsius
        self.v_init = v_init  # holding potential
        h.dt = h_dt  # ms - value of the fundamental integration time step, dt, used by fadvance().
        self.dur = dur  # clamp duration, ms
        self.step = step  # voltage clamp increment, the user can
        self.st_cl = st_cl  # clamp start, mV
        self.end_cl = end_cl  # clamp end, mV
        self.v_cl = v_cl  # actual voltage clamp, mV

        # a two-electrodes voltage clamp
        self.f3cl = h.VClamp(self.soma(0.5))
        self.f3cl.dur[0] = f3cl_dur0  # ms
        self.f3cl.amp[0] = f3cl_amp0  # mV
        self.f3cl.dur[1] = dur  # ms
        self.f3cl.amp[1] = v_cl  # mV
        self.f3cl.dur[2] = f3cl_dur2  # ms
        self.f3cl.amp[2] = f3cl_amp2  # mV

        # vectors for data handling
        self.t_vec = []  # vector for time steps (h.dt)
        self.v_vec = np.arange(st_cl, end_cl, step)  # vector for voltage
        self.v_vec_t = []  # vector for voltage as function of time
        self.i_vec = []  # vector for current
        self.ipeak_vec = []  # vector for peak current
        self.gnorm_vec = []  # vector for normalized conductance
        self.all_is = []  # all currents
        self.all_v_vec_t = []

        self.L = len(self.v_vec)

        # conductance attributes for plotting
        self.vrev = 0
        self.v_half = 0
        self.s = 0

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
                # append data
                self.t_vec.append(h.t)
                self.v_vec_t.append(self.soma.v)
                self.i_vec.append(dens)
                # advance
                h.fadvance()

        # find i peak of trace
        self.ipeak_vec.append(self.find_ipeaks())

    def clamp_at_volt(self, v_cl):
        """ Runs a trace and calculates peak currents.
        Args:
            v_cl (int): voltage to run
        """
        if self.gnorm_vec == []:
            time_padding = 5  # ms
            h.tstop = time_padding + self.dur + time_padding  # time stop
            
        curr_tr = 0  # initialization of peak current
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.
        pre_i = 0  # initialization of variables used to commute the peak current
        dens = 0
        self.f3cl.amp[1] = v_cl  # mV
        for _ in self.ntrials:
            while h.t < h.tstop:  # runs a single trace, calculates peak current
                dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                    0.5).i_cap  # clamping current in mA/cm2, for each dt
                # append data
                self.t_vec.append(h.t)
                self.v_vec_t.append(self.soma.v)
                self.i_vec.append(dens)
                # advance
                h.fadvance()

        # find i peak of trace
        self.ipeak_vec.append(self.find_ipeaks())

    def find_ipeaks(self):
        """
        Evaluate the peak and updates the peak current.
        Returns peak current.
        Finds positive and negative peaks.
        """
        self.i_vec = np.array(self.i_vec)
        self.t_vec = np.array(self.t_vec)
        mask = np.where(np.logical_and(self.t_vec >= 4, self.t_vec <= 10))
        i_slice = self.i_vec[mask]
        curr_max = np.max(i_slice)
        curr_min = np.min(i_slice)
        if np.abs(curr_max) > np.abs(curr_min):
            curr_tr = curr_max
        else:
            curr_tr = curr_min
        return curr_tr
    
    def find_ipeaks_with_index(self):
        """
        Evaluate the peak and updates the peak current.
        Returns peak current.
        Finds positive and negative peaks.
        """
        self.i_vec = np.array(self.i_vec)
        self.t_vec = np.array(self.t_vec)
        mask = np.where(np.logical_and(self.t_vec >= 4, self.t_vec <= 10))
        i_slice = self.i_vec[mask]
        curr_max = np.max(i_slice)
        curr_min = np.min(i_slice)
        if np.abs(curr_max) > np.abs(curr_min):
            curr_tr = curr_max
        else:
            curr_tr = curr_min
        curr_tr_index = list(i_slice).index(curr_tr)
        return curr_tr_index, curr_tr

    def findG(self, v_vec, ipeak_vec):
        """ Returns normalized conductance vector
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

        # boltzmann for conductance
        def boltzmann(vm, Gmax, v_half, s):
            return Gmax * (vm - self.vrev) / (1 + np.exp((v_half - vm) / s))

        self.vrev = stats.linregress(lin_i, lin_v).intercept
        Gmax, self.v_half, self.s = optimize.curve_fit(boltzmann, v_vec, ipeak_vec)[0]

        # find normalized conductances at each voltage
        norm_g = h.Vector()
        for volt in v_vec:
            norm_g.append(1 / (1 + np.exp(-(volt - self.v_half) / self.s)))
        return norm_g

    def genActivation(self):
        """ Generates simulated activation data
        Returns:
            gnorm_vec: normalized peak conductance vector
            voltages
            all_is: peak current vector
        """
        if self.gnorm_vec == []:
            time_padding = 5  # ms
            h.tstop = time_padding + self.dur + time_padding  # time stop

            # iterates across voltages (mV)
            for v_cl in np.arange(self.st_cl, self.end_cl, self.step):  # self.vec
                # resizing the vectors
                self.t_vec = []
                self.i_vec = []
                self.v_vec_t = []

                self.clamp(v_cl)

                self.all_is.append(self.i_vec[1:])
                self.all_v_vec_t.append(self.v_vec_t)

            # calculate normalized peak conductance
            self.gnorm_vec = self.findG(self.v_vec, self.ipeak_vec)

        return self.gnorm_vec, self.v_vec, self.all_is

    def plotActivation_VGnorm(self):
        """
        Saves activation plot as PGN file.
        """
        plt.figure()
        plt.xlabel('Voltage $(mV)$')
        plt.ylabel('Normalized conductance')
        plt.title('Activation: Voltage/Normalized conductance')
        plt.plot(self.v_vec, self.gnorm_vec, 'o', c='black')
        gv_slope, v_half, top, bottom = cf.calc_act_obj(self.channel_name)
        formatted_gv_slope = np.round(gv_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5, f'Slope: {formatted_gv_slope}')
        plt.text(-10, 0.3, f'V50: {formatted_v_half}')
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c='red')
        # save as PGN file
        plt.savefig(
            os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Activation Voltage-Normalized Conductance Relation'))
    
    def plotActivation_VGnorm_plt(self,plt,color):
        """
        Saves activation plot as PGN file.
        """
        
        diff = 0
        if color == 'red':
            diff = 0.5 
        
        
        plt.plot(self.v_vec, self.gnorm_vec, 'o', c=color)
        gv_slope, v_half, top, bottom = cf.calc_act_obj(self.channel_name)
        formatted_gv_slope = np.round(gv_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5 + diff, f'Slope: {formatted_gv_slope}', c = color)
        plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c = color)
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c=color)
        return (formatted_v_half, formatted_gv_slope)
        
        
    def plotActivation_IVCurve(self):
        plt.figure()
        plt.xlabel('Voltage $(mV)$')
        plt.ylabel('Peak Current $(pA)$')
        plt.title("Activation: IV Curve")
        plt.plot(np.array(self.v_vec), np.array(self.ipeak_vec), 'o', c='black')
        #plt.text(-110, -0.05, 'Vrev at ' + str(round(self.vrev, 1)) + ' mV', fontsize=10, c='blue')
        formatted_peak_i = np.round(min(self.ipeak_vec), decimals=2)
        #plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Activation IV Curve"))
    
    def plotActivation_IVCurve_plt(self,plt,color):
        
        plt.plot(np.array(self.v_vec), np.array(self.ipeak_vec), 'o', c=color)
        #plt.text(-110, -0.05, 'Vrev at ' + str(round(self.vrev, 1)) + ' mV', fontsize=10, c='blue')
        formatted_peak_i = np.round(min(self.ipeak_vec), decimals=2)
        #plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps
        
    def plotActivation_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('Activation Time/Voltage relation')
        [plt.plot(self.t_vec, self.all_v_vec_t[i], c='black') for i in np.arange(self.L)]

    def plotActivation_TimeVRelation_plt(self,plt,color):
        [plt.plot(self.t_vec, self.all_v_vec_t[i], c=color) for i in np.arange(self.L)]
        
    def plotActivation_TCurrDensityRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('Activation Time/Current density relation')
        curr = np.array(self.all_is)
        mask = np.where(np.logical_or(self.v_vec == -50, self.v_vec == -60))
        [plt.plot(self.t_vec[1:], curr[i], c='black') for i in np.arange(len(curr))[mask]]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Activation Time Current Density Relation"))

    def plotActivation_TCurrDensityRelation_plt(self,plt,color):
        curr = np.array(self.all_is)
        mask = np.where(np.logical_or(self.v_vec == -50, self.v_vec == -60))
        [plt.plot(self.t_vec[1:], curr[i], c=color) for i in np.arange(len(curr))[mask]]
        
    def plotActivation_allTraces(self):
        curr = np.array(self.all_is)
        for volt in self.v_vec:
            plt.figure()
            plt.xlabel('Time $(ms)$')
            plt.ylabel('Current density $(mA/cm^2)$')
            plt.title(f"Activation Traces for {volt} mV")
            mask = np.where(self.v_vec == volt)
            plt.plot(self.t_vec[1:], curr[mask][0], c='black')
            # save as PGN file
            plt.savefig(os.path.join(os.path.split(__file__)[0], f"Plots_Folder/Activation Traces for {volt} mV"))

    def plotAllActivation(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotActivation_VGnorm()
        self.plotActivation_IVCurve()
        self.plotActivation_TimeVRelation()
        self.plotActivation_TCurrDensityRelation()
        #self.plotActivation_allTraces()


##################
# Inactivation
##################
class Inactivation:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.025, ntrials=range(30),
                 dur=500, step=5, st_cl=-120, end_cl=40, v_cl=-120,
                 f3cl_dur0=40, f3cl_amp0=-120, f3cl_dur2=20, f3cl_amp2=-10):

        self.h = h  # NEURON h

        # one-compartment cell (soma)
        self.channel_name = channel_name
        self.soma = h.Section(name='soma2')
        self.soma.diam = soma_diam  # micron
        self.soma.L = soma_L  # micron, so that area = 10000 micron2
        self.soma.nseg = soma_nseg  # adimensional
        self.soma.cm = soma_cm  # uF/cm2
        self.soma.Ra = soma_Ra  # ohm-cm
        self.soma.insert(channel_name)  # insert mechanism
        self.soma.ena = soma_ena

        # clamping parameters
        self.ntrials = ntrials  #
        h.celsius = h_celsius  # temperature in celsius
        self.v_init = v_init  # holding potential
        h.dt = h_dt  # ms - value of the fundamental integration time step, dt, used by fadvance().
        self.dur = dur  # clamp duration, ms
        self.step = step  # voltage clamp increment, the user can
        self.st_cl = st_cl  # clamp start, mV
        self.end_cl = end_cl  # clamp end, mV
        self.v_cl = v_cl  # actual voltage clamp, mV

        # a two-electrodes voltage clamp
        self.f3cl = h.VClamp(self.soma(0.5))
        self.f3cl.dur[0] = f3cl_dur0  # ms
        self.f3cl.amp[0] = f3cl_amp0  # mV
        self.f3cl.dur[1] = dur  # ms
        self.f3cl.amp[1] = v_cl  # mV
        self.f3cl.dur[2] = f3cl_dur2  # ms
        self.f3cl.amp[2] = f3cl_amp2  # mV

        # vectors for data handling
        self.t_vec = []  # vector for time steps (h.dt)
        self.v_vec = np.arange(st_cl, end_cl, step)  # vector for voltage
        self.v_vec_t = []  # vector for voltage as function of time
        self.i_vec = []  # vector for current
        self.ipeak_vec = []  # vector for peak current
        self.inorm_vec = []  # vector for normalized current
        self.all_is = []  # all currents
        self.all_v_vec_t = []  # all voltages

        self.L = len(self.v_vec)

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
                dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                    0.5).i_cap  # clamping current in mA/cm2, for each dt

                self.t_vec.append(h.t)  # code for store the current
                self.v_vec_t.append(self.soma.v)  # trace to be plotted
                self.i_vec.append(dens)  # trace to be plotted

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
                dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                    0.5).i_cap  # clamping current in mA/cm2, for each dt

                self.t_vec.append(h.t)  # code for store the current
                self.v_vec_t.append(self.soma.v)  # trace to be plotted
                self.i_vec.append(dens)  # trace to be plotted

                h.fadvance()

        # find i peak of trace
        self.ipeak_vec.append(self.find_ipeaks())

    def find_ipeaks(self):
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
        else:
            peak_curr = i_slice[peak_indices][0]
        return peak_curr
    
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

    def genInactivation(self):
        if self.inorm_vec == []:
            h.tstop = 40 + self.dur + 20  # TODO fix padding

            for v_cl in self.v_vec:  # iterates across voltages
                # resizing the vectors
                self.t_vec = []
                self.i_vec = []
                self.v_vec_t = []

                self.clamp(v_cl)
                self.all_is.append(self.i_vec[1:])
                self.all_v_vec_t.append(self.v_vec_t)

            # normalization of peak current with respect to the min since the values are negative
            ipeak_min = min(self.ipeak_vec)
            self.inorm_vec = np.array(self.ipeak_vec) / ipeak_min

        return self.inorm_vec, self.v_vec, self.all_is

    def plotInactivation_VInormRelation(self):
        plt.figure()
        plt.xlabel('Voltage $(mV)$')
        plt.ylabel('Normalized current')
        plt.title('Inactivation: Voltage/Normalized Current Relation')
        plt.plot(self.v_vec, self.inorm_vec, 'o', c='black')
        ssi_slope, v_half, top, bottom, tau0 = cf.calc_inact_obj(self.channel_name)
        formatted_ssi_slope = np.round(ssi_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        #plt.text(-10, 0.5, f'Slope: {formatted_ssi_slope}')
        #plt.text(-10, 0.3, f'V50: {formatted_v_half}')
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, ssi_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c='red')
        # save as PGN file
        plt.savefig(
            os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Inactivation Voltage Normalized Current Relation'))
        
    def plotInactivation_VInormRelation_plt(self, plt, color):
        
        diff = 0
        if color == 'red':
            diff = 0.5
        #plt.plot(self.v_vec, self.inorm_vec, 'o', c='black')
        ssi_slope, v_half, top, bottom, tau0 = cf.calc_inact_obj(self.channel_name)
        formatted_ssi_slope = np.round(ssi_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5 + diff, f'Slope: {formatted_ssi_slope}', c = color)
        plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c = color)
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, ssi_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c=color)
        return (formatted_v_half, formatted_ssi_slope)

    def plotInactivation_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('Inactivation Time/Voltage relation')
        [plt.plot(self.t_vec, self.all_v_vec_t[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Inactivation Time Voltage Relation'))
        
    def plotInactivation_TimeVRelation_plt(self, plt, color):
        [plt.plot(self.t_vec, self.all_v_vec_t[i], c=color) for i in np.arange(self.L)]

    def plotInactivation_TCurrDensityRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('Inactivation Time/Current density relation')
        [plt.plot(self.t_vec[1:], self.all_is[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Inactivation Time Current Density Relation"))
    
    
    def plotInactivation_TCurrDensityRelation(self, plt,color):
        [plt.plot(self.t_vec[1:], self.all_is[i], c=color) for i in np.arange(self.L)]

    def plotInactivation_Tau_0mV(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('Inactivation Tau at 0 mV')
        # select 0 mV
        volt = 0  # mV
        mask = np.where(self.v_vec == volt)[0]
        curr = np.array(self.all_is)[mask][0]
        time = np.array(self.t_vec)[1:]
        # fit exp: IFit(t) = A * exp (-t/τ) + C
        ts, data, xs, ys, tau = self.find_tau0_inact(curr)
        # plot
        plt.plot(ts, data, color="black")
        plt.plot(xs, ys, color="red")
        formatted_tau = np.round(tau, decimals=3)
        #plt.text(0.2, -0.01, f"Tau at 0 mV: {formatted_tau}", color="blue")
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Inactivation Tau at 0 mV"))
    
    
    def get_just_tau0(self):
        try:
            volt = 0  # mV
            mask = np.where(self.v_vec == volt)[0]
            curr = np.array(self.all_is)[mask][0]
            time = np.array(self.t_vec)[1:]
            # fit exp: IFit(t) = A * exp (-t/τ) + C
            ts, data, xs, ys, tau = self.find_tau0_inact(curr)
            return tau
        except:
            return 9999999999 # return very bad tau if cannot be fit
        

    def plotInactivation_Tau_0mV_plt(self, plt,color, upper = 700):
        
        diff = 0
        if color == 'red':
            diff = 1.5
            
        def fit_expon(x, a, b, c):
            return a + b * np.exp(-1 * c * x)
        act = ggsd.Activation(channel_name = 'na16')
        act.clamp_at_volt(0)
        starting_index = list(act.i_vec).index(act.find_ipeaks_with_index()[1])

        t_vecc = act.t_vec[starting_index:upper]
        i_vecc = act.i_vec[starting_index:upper]
        popt, pcov = optimize.curve_fit(fit_expon,t_vecc,i_vecc, method = 'dogbox')
        
        tau = popt[2]
        tau = 1000 * tau

        fitted_i = fit_expon(act.t_vec[starting_index:upper],popt[0],popt[1],popt[2])
        plt.plot(act.t_vec[starting_index:upper], fitted_i, c=color)
        plt.text(0.2, -2 + diff, f"Tau at 0 mV: {tau}", color=color)

        return tau
            

        # select 0 mV
        volt = 0  # mV
        mask = np.where(self.v_vec == volt)[0]
        curr = np.array(self.all_is)[mask][0]
        time = np.array(self.t_vec)[1:]
        # fit exp: IFit(t) = A * exp (-t/τ) + C
        ts, data, xs, ys, tau = self.find_tau0_inact(curr)
        # plot
        plt.plot(ts, data, color=color)
        plt.plot(xs, ys, color=color)
        formatted_tau0 = np.round(tau, decimals=3)
        
        return tau
        
        
    def fit_exp(self, x, a, b, c):
        """
        IFit(t) = A * exp (-t/τ) + C
        """
        return a * np.exp(-x / b) + c
    
    def one_phase(self, x, y0, plateau, k):
        '''
        Fit a one-phase association curve to an array of data points X. 
        For info about the parameters, visit 
        https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_exponential_association.htm    
        '''
        return y0 + (plateau - y0) * (1 - np.exp(-k * x))
    
    # one phase asso
    # 1/b as tau

    def find_tau0_inact(self, raw_data):
        
        
        def one_phase(x, y0, plateau, k):
            return y0 + (plateau - y0) * (1 - np.exp(-k * x))
        
        def fit_expon(x, a, b, c):
            return a + b * np.exp(-1 * c * x)
    
    
        # take peak curr and onwards
        min_val, mindex = min((val, idx) for (idx, val) in enumerate(raw_data[:int(0.7 * len(raw_data))]))
        padding = 15  # after peak
        data = raw_data[mindex:mindex + padding]
        ts = [0.1 * i for i in range(len(data))]  # make x values which match sample times

        # calc tau and fit exp
        # cuts data points in half
        length = len(ts) // 2
        popt, pcov = optimize.curve_fit(fit_expon, ts[0:length], data[0:length])  # fit exponential curve
        perr = np.sqrt(np.diag(pcov))
        # print('in ' + str(all_tau_sweeps[i]) + ' the error was ' + str(perr))
        xs = np.linspace(ts[0], ts[len(ts) - 1], 1000)  # create uniform x values to graph curve
        ys = fit_expon(xs, *popt)  # get y values
        vmax = max(ys) - min(ys)  # get diff of max and min voltage
        vt = min(ys) + .37 * vmax  # get vmax*1/e
        tau = popt[2]
        return ts, data, xs, ys, tau

    def plotAllInactivation(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotInactivation_VInormRelation()
        self.plotInactivation_TimeVRelation()
        self.plotInactivation_TCurrDensityRelation()
        self.plotInactivation_Tau_0mV()


##################
# Recovery from Inactivation (RFI)
# &  RFI Tau
##################
class RFI:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-90, h_dt=0.1, ntrials=30,
                 min_inter=0.1, max_inter=5000, num_pts=50, cond_st_dur=1000, res_pot=-90, dur=0.1,
                 vec_pts=[1, 1.5, 3, 5.6, 10, 30, 56, 100, 150, 300, 560, 1000, 2930, 5000],
                 f3cl_dur0=5, f3cl_amp0=-90, f3cl_amp1=0, f3cl_dur3=20, f3cl_amp3=0, f3cl_dur4=5, f3cl_amp4=-90):

        self.h = h  # NEURON h

        # one-compartment cell (soma)
        self.channel_name = channel_name
        self.soma = h.Section(name='soma2')
        self.soma.diam = soma_diam  # micron
        self.soma.L = soma_L  # micron, so that area = 10000 micron2
        self.soma.nseg = soma_nseg  # adimensional
        self.soma.cm = soma_cm  # uF/cm2
        self.soma.Ra = soma_Ra  # ohm-cm
        self.soma.insert(channel_name)  # insert mechanism
        self.soma.ena = soma_ena

        # clamping parameters
        self.ntrials = ntrials  #
        h.celsius = h_celsius  # temperature in celsius
        self.v_init = v_init  # holding potential
        h.dt = h_dt  # ms - value of the fundamental integration time step, dt, used by fadvance().
        self.dur = dur  # clamp duration, ms
        self.min_inter = min_inter  # pre-stimulus starting interval # TODO use?
        self.max_inter = max_inter  # pre-stimulus endinging interval
        self.num_pts = num_pts  # number of points in logaritmic scale

        self.cond_st_dur = cond_st_dur  # conditioning stimulus duration
        self.res_pot = res_pot  # resting potential

        # vector containing 'num_pts' values equispaced between log10(min_inter) and log10(max_inter)
        # for RecInactTau
        self.vec_pts = vec_pts
        # vec_pts = np.logspace(np.log10(min_inter), np.log10(max_inter), num=num_pts)

        # voltage clamp with "five" levels for RecInactTau
        self.f3cl = h.VClamp_plus(self.soma(0.5))
        self.f3cl.dur[0] = f3cl_dur0  # ms
        self.f3cl.amp[0] = f3cl_amp0  # mV
        self.f3cl.dur[1] = cond_st_dur  # ms default 1000
        self.f3cl.amp[1] = f3cl_amp1  # mV
        self.f3cl.dur[2] = dur  # ms
        self.f3cl.amp[2] = res_pot  # mV default -120
        self.f3cl.dur[3] = f3cl_dur3  # ms
        self.f3cl.amp[3] = f3cl_amp3  # mV
        self.f3cl.dur[4] = f3cl_dur4  # ms
        self.f3cl.amp[4] = f3cl_amp4  # mV

        # vectors for data handling RecInactTau
        self.rec_vec = []  # RFI (peak2/peak1)
        self.time_vec = []  # same as time in vec_pts
        self.log_time_vec = []  # same as time in vec_ptsm but logged
        self.t_vec = []  # vector for time steps (h.dt)
        self.v_vec_t = []  # vector for voltage as function of time
        self.i_vec_t = []  # vector for current
        self.rec_inact_tau_vec = []  # RFI taus
        self.all_is = []  # all currents
        self.all_v_vec_t = []  # all voltages
        self.all_t_vec = []  # all h.t

        self.L = len(self.vec_pts)

    def clampRecInactTau(self, dur):
        """ Runs a trace and calculates peak currents.
        Args:
            dur (int): duration (ms) to run
        """
        self.f3cl.dur[2] = dur
        h.tstop = 5 + 1000 + dur + 20 + 5  # TODO fix padding
        h.finitialize(self.v_init)

        # variables initialization
        pre_i1 = 0
        pre_i2 = 0
        peak_curr1 = 0
        peak_curr2 = 0

        while h.t < h.tstop:  # runs a single trace, calculates peak current

            dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                0.5).i_cap  # clamping current in mA/cm2, for each dt
            self.t_vec.append(h.t)
            self.v_vec_t.append(self.soma.v)
            self.i_vec_t.append(dens)

            if (h.t > 5) and (h.t < 15):  # evaluate the first peak
                if pre_i1 < abs(dens):
                    peak_curr1 = abs(dens)
                pre_i1 = abs(dens)

            if (h.t > (5 + self.cond_st_dur + dur)) and (
                    h.t < (15 + self.cond_st_dur + dur)):  # evaluate the second peak

                if pre_i2 < abs(dens):
                    peak_curr2 = abs(dens)
                pre_i2 = abs(dens)

            h.fadvance()

        # updates the vectors at the end of the run
        self.time_vec.append(dur)
        self.log_time_vec.append(np.log10(dur))
        peak_curr1 = self.find_ipeaks(start_ht=5, end_ht=15)
        peak_curr2 = self.find_ipeaks(start_ht=5+self.cond_st_dur + dur, end_ht=15 + self.cond_st_dur + dur)
        self.rec_vec.append(peak_curr2 / peak_curr1)

        # calc tau using RF and tstop
        # append values to vector
        RF_t = peak_curr2 / peak_curr1
        tau = -h.tstop / np.log(-RF_t + 1)
        self.rec_inact_tau_vec.append(tau)

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

    def genRecInactTau(self):
        recov = []  # RFI tau curve
        for dur in self.vec_pts:
            # resizing the vectors
            self.t_vec = []
            self.i_vec_t = []
            self.v_vec_t = []

            self.clampRecInactTau(dur)
            recov.append(self.rec_vec)
            self.all_is.append(self.i_vec_t)
            self.all_v_vec_t.append(self.v_vec_t)
            self.all_t_vec.append(self.t_vec)

        return self.rec_inact_tau_vec, recov, self.vec_pts

    def plotRFI_LogVInormRelation(self):
        plt.figure()
        plt.xlabel('Log(Time)')
        plt.ylabel('Fractional recovery (P2/P1)')
        plt.title('Log(Time)/Fractional recovery (P2/P1)')
        plt.plot(self.log_time_vec, self.rec_vec, 'o', c='black')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/RFI Log Time Fractional recovery Relation'))

    def plotRFI_VInormRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Fractional recovery (P2/P1)')
        plt.title('Time/Fractional recovery (P2/P1)')
        y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(self.channel_name)
        formatted_tauSlow = np.round(1 / k_slow, decimals=2)
        formatted_tauFast = np.round(1 / k_fast, decimals=2)
        formatted_percentFast = np.round(percent_fast, decimals=4)
        plt.text(-10, 0.75, f'Tau Slow: {formatted_tauSlow}')
        plt.text(-10, 0.8, f'Tau Fast: {formatted_tauFast}')
        plt.text(-10, 0.85, f'% Fast Component: {formatted_percentFast}')
        plt.plot(self.time_vec, self.rec_vec, 'o', c='black')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/RFI Time Fractional recovery Relation'))

    def plotRFI_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('RFI Time/Voltage relation')
        [plt.plot(self.all_t_vec[i], self.all_v_vec_t[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/RFI Time Voltage Relation'))

    def plotRFI_TCurrDensityRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('RFI Time/Current density relation')
        [plt.plot(self.all_t_vec[i], self.all_is[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/RFI Time Current Density Relation"))

    def plotAllRFI(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotRFI_VInormRelation()
        self.plotRFI_LogVInormRelation()
        self.plotRFI_TimeVRelation()
        self.plotRFI_TCurrDensityRelation()

    def plotAllRFI_with_ax(self, fig_title,
                           figsize=(18, 9), color='black',
                           saveAsFileName=None, loadFileName=None, saveAsPNGFileName=None):
        """
        Creates new ax if loadFileName is not a valid string. Plots all.
        color = 'red' if cur_params == "variant_params" else 'black'
        color = red if HMM else black
        """
        if loadFileName:
            # read pkl file
            with open(loadFileName, 'rb') as fid:
                ax = pickle.load(fid)
        else:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
            fig.suptitle(fig_title, fontsize=15, fontweight='bold')
            fig.subplots_adjust(wspace=0.5)
            fig.subplots_adjust(hspace=0.5)

        y_offset = -0.2 if color == "red" else 0
        label = 'HH' if color == 'black' else 'HMM'

        # upper left
        ax[0, 0].set_xlabel('Time $(ms)$')
        ax[0, 0].set_ylabel('Fractional recovery (P2/P1)')
        ax[0, 0].set_title('Time/Fractional recovery (P2/P1)')
        y0, plateau, percent_fast, k_fast, k_slow = cf.calc_recov_obj(self.channel_name)
        formatted_tauSlow = np.round(1 / k_slow, decimals=2)
        formatted_tauFast = np.round(1 / k_fast, decimals=2)
        formatted_percentFast = np.round(percent_fast, decimals=4)
        ax[0, 0].text(-8, 0.65 + y_offset, f'Tau Slow: {formatted_tauSlow}', c=color)
        ax[0, 0].text(-8, 0.70 + y_offset, f'Tau Fast: {formatted_tauFast}', c=color)
        ax[0, 0].text(-8, 0.75 + y_offset, f'% Fast Component: {formatted_percentFast}', c=color)
        ax[0, 0].plot(self.time_vec, self.rec_vec, 'o', c=color, label=label)
        ax[0, 0].legend(loc='lower right')  # add legend

        # lower left
        ax[1, 0].set_xlabel('Log(Time)')
        ax[1, 0].set_ylabel('Fractional recovery (P2/P1)')
        ax[1, 0].set_title('Log(Time)/Fractional recovery (P2/P1)')
        ax[1, 0].plot(self.log_time_vec, self.rec_vec, 'o', c=color)

        # upper right
        ax[0, 1].set_xlabel('Time $(ms)$')
        ax[0, 1].set_ylabel('Voltage $(mV)$')
        ax[0, 1].set_title('RFI Time/Voltage relation')
        [ax[0, 1].plot(self.all_t_vec[i], self.all_v_vec_t[i], c=color) for i in np.arange(self.L)]

        # lower right
        ax[1, 1].set_xlabel('Time $(ms)$')
        ax[1, 1].set_ylabel('Current density $(mA/cm^2)$')
        ax[1, 1].set_title('RFI Time/Current density relation')
        [ax[1, 1].plot(self.all_t_vec[i], self.all_is[i], c=color) for i in np.arange(self.L)]

        if saveAsFileName:
            with open(saveAsFileName, 'wb') as fid:
                pickle.dump(ax, fid)
        if saveAsPNGFileName:
            plt.savefig(os.path.join(os.path.split(__file__)[0], saveAsPNGFileName))


##################
# Ramp Protocol
##################
class Ramp:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, t_init=30,
                 v_first_step=-60, t_first_step=30, v_ramp_end=0, t_ramp=300, t_plateau=100,
                 v_last_step=-120, t_last_step=30, h_dt=0.025):
        self.h = h  # NEURON h
        # one-compartment cell (soma)
        self.soma = h.Section(name='soma2')
        self.soma.diam = soma_diam  # micron
        self.soma.L = soma_L  # micron, so that area = 10000 micron2
        self.soma.nseg = soma_nseg  # adimensional
        self.soma.cm = soma_cm  # uF/cm2
        self.soma.Ra = soma_Ra  # ohm-cm
        self.soma.insert(channel_name)  # insert mechanism
        self.soma.ena = soma_ena

        self.time_steps_arr = []

        # clamping parameters
        def make_ramp():
            time_steps_arr = np.array([t_init, t_first_step, t_ramp, t_plateau, t_last_step])
            time_steps_arr = (time_steps_arr / h_dt).astype(int)
            time_steps_arr = np.cumsum(time_steps_arr)
            ntimesteps = time_steps_arr[-1]
            ramp_v = np.zeros(ntimesteps)
            ramp_v[0:time_steps_arr[0]] = v_init
            ramp_v[time_steps_arr[0]:time_steps_arr[1]] = v_first_step
            ramp_v[time_steps_arr[1]:time_steps_arr[2]] = np.linspace(v_first_step, v_ramp_end,
                                                                      time_steps_arr[2] - time_steps_arr[1])
            ramp_v[time_steps_arr[2]:time_steps_arr[3]] = v_ramp_end
            ramp_v[time_steps_arr[3]:time_steps_arr[4]] = v_last_step
            self.time_steps_arr = time_steps_arr
            return ramp_v

        self.ntrials = 1  #
        h.celsius = h_celsius  # temperature in celsius
        self.stim_ramp = make_ramp()  # the voltage of the whole protocol
        h.dt = h_dt  # ms - value of the fundamental integration time step, dt, used by fadvance().
        self.v_init = v_init  # holding potential
        self.t_start_persist = int((t_init + t_first_step + t_ramp) / h_dt)  # time that plateau starts
        self.t_end_persist = int((t_init + t_first_step + t_ramp + t_plateau) / h_dt)  # time that plateau ends
        self.t_total = t_init + t_first_step + t_ramp + t_plateau + t_last_step

        # a two-electrodes voltage clamp
        self.f3cl = h.VClamp(self.soma(0.5))
        self.f3cl.dur[0] = 1e9
        self.f3cl.amp[0] = self.stim_ramp[0]

        # vectors for data handling
        self.t_vec = np.ones(len(self.stim_ramp)) * h_dt
        self.t_vec = np.cumsum(self.t_vec)
        self.v_vec = self.stim_ramp
        self.v_vec_t = []  # vector for voltage as function of time
        self.i_vec = []  # vector for current

    def clamp(self, v_cl):
        """ Runs a trace and calculates currents.
        Args:
            v_cl (int): voltage to run
        """
        self.f3cl.amp[0] = v_cl
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.
        # parameters initialization
        stim_counter = 0

        dtsave = h.dt
        while round(h.t, 3) < h.tstop:  # runs a single trace, calculates current
            self.f3cl.amp[0] = self.stim_ramp[stim_counter]
            dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                0.5).i_cap  # clamping current in mA/cm2, for each dt

            self.v_vec_t.append(self.soma.v)  # trace to be plotted
            self.i_vec.append(dens)  # trace to be plotted

            stim_counter += 1
            h.fadvance()

    def genRamp(self):
        h.tstop = self.t_total
        self.clamp(self.v_vec[0])

    def areaUnderCurve(self):
        """ Calculates and returns normalized area (to activation IV) under IV curve of Ramp
        """
        maskStart, maskEnd = self.time_steps_arr[1], self.time_steps_arr[2]  # selects ramp (incline) portion only
        i_vec_ramp = self.i_vec[maskStart:maskEnd]
        v_vec_t_ramp = self.v_vec_t[maskStart:maskEnd]
        # plt.plot(self.t_vec[maskStart:maskEnd], self.v_vec[maskStart:maskEnd], color= 'b') # uncomment to view area taken
        area = trapz(i_vec_ramp, x=v_vec_t_ramp)  # find area
        act = Activation(channel_name='na16')
        act.genActivation()
        area = area / min(act.ipeak_vec)  # normalize to peak currents from activation
        return area

    def persistentCurrent(self):
        """ Calculates persistent current (avg current of last 100 ms at 0 mV)
        Normalized by peak from IV (same number as areaUnderCurve).
        """
        persistent = self.i_vec[self.t_start_persist:self.t_end_persist]
        act = Activation()
        act.genActivation()
        IVPeak = min(act.ipeak_vec)
        return (sum(persistent) / len(persistent)) / IVPeak

    def plotRamp_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('Ramp Time/Voltage relation')
        plt.plot(self.t_vec, self.v_vec, color='black')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Ramp Time Voltage relation'))

    def plotRamp_TimeCurrentRelation(self):
        area = round(self.areaUnderCurve(), 2)
        persistCurr = "{:.2e}".format(round(self.persistentCurrent(), 4))

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        f.add_subplot(111, frameon=False)  # for shared axes labels and big title
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.title("Ramp: Time Current Density Relation", x=0.4, y=1.1)
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current', labelpad=25)

        # starting + first step + ramp section
        ax1.set_title("Ramp")
        ax1.plot(self.t_vec[1:self.t_start_persist], self.i_vec[1:self.t_start_persist], 'o', c='black', markersize=0.1)
        plt.text(0.05, 0.2, f'Normalized \narea under \ncurve: {area}', c='blue', fontsize=10)

        # persistent current + last step section
        ax2.set_title("Persistent Current")
        ax2.plot(self.t_vec[self.t_start_persist:], self.i_vec[self.t_start_persist:], 'o', c='black', markersize=0.1)
        plt.text(0.75, 0.5, f'Persistent Current:\n{persistCurr} mV', c='blue', fontsize=10, ha='center')

        # save as PGN file
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Ramp Time Current Density Relation'))

    def plotAllRamp(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotRamp_TimeVRelation()
        self.plotRamp_TimeCurrentRelation()
        
    def plotAllRamp_with_ax(self, ax_list, cur_params):
        color = 'red' if cur_params == "variant_params" else 'black'
        y_offset = -0.02 if cur_params == "variant_params" else 0
        x_offset = 0

        ax_list[0].set_xlabel('Time $(ms)$')
        ax_list[0].set_ylabel('Voltage $(mV)$')
        ax_list[0].set_title('Ramp Time/Voltage relation')
        ax_list[0].plot(self.t_vec, self.v_vec, color=color)

        area = round(self.areaUnderCurve(), 2)
        persistCurr = "{:.2e}".format(round(self.persistentCurrent(), 4))

        # starting + first step + ramp section
        ax_list[1].set_xlabel('Time $(ms)$')
        ax_list[1].set_ylabel('Current', labelpad=25)
        ax_list[1].set_title("Ramp")
        ax_list[1].plot(self.t_vec[1:self.t_start_persist], self.i_vec[1:self.t_start_persist], 'o', c=color,
                        markersize=0.1)
        ax_list[1].text(0 + x_offset, -0.08 + y_offset, f'Normalized \narea under \ncurve: {area}', color=color,
                        fontsize=10)

        # persistent current + last step section
        ax_list[2].set_xlabel('Time $(ms)$')
        ax_list[2].set_ylabel('Current', labelpad=25)
        ax_list[2].set_title("Persistent Current")
        ax_list[2].plot(self.t_vec[self.t_start_persist:], self.i_vec[self.t_start_persist:], 'o', c=color,
                        markersize=0.1)
        ax_list[2].text(420 + x_offset, -0.04 + y_offset, f'Persistent Current:\n{persistCurr} mV', color=color,
                        fontsize=10, ha='center')


##################
# UDB20 Protocol
##################
class UDB20:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, v_held=-70,
                 v_peak=-10, t_peakdur=100, t_init=200, num_repeats=9, h_dt=0.025):
        self.h = h  # NEURON h
        # one-compartment cell (soma)
        self.soma = h.Section(name='soma2')
        self.soma.diam = soma_diam  # micron
        self.soma.L = soma_L  # micron, so that area = 10000 micron2
        self.soma.nseg = soma_nseg  # adimensional
        self.soma.cm = soma_cm  # uF/cm2
        self.soma.Ra = soma_Ra  # ohm-cm
        self.soma.insert(channel_name)  # insert mechanism
        self.soma.ena = soma_ena

        # clamping parameters
        self.num_repeats = num_repeats  # number of iterations
        h.celsius = h_celsius  # temperature in celsius
        h.dt = h_dt  # ms - value of the fundamental integration time step, dt, used by fadvance()
        self.v_init = v_init
        self.t_init = t_init
        self.v_held = v_held  # held potential between peaks (mV)
        self.v_peak = v_peak  # membrane potential of peak (mV)
        self.t_peakdur = t_peakdur  # duration of each peak (ms)
        self.t_total = t_init + (t_peakdur * num_repeats * 2)  # total time of protocol

        # a two-electrodes voltage clamp
        self.f3cl = h.VClamp(self.soma(0.5))
        self.f3cl.dur[0] = 1e9
        self.f3cl.amp[0] = v_init

        def make_UDB():
            # creates time and voltage vectors for UDB20 protocol
            time_steps = np.arange(0, self.t_total, h.dt)
            UDB_v = np.zeros(len(time_steps))
            UDB_v[0:int(t_init / h.dt)] = self.v_init

            stim_len = int(t_peakdur / h.dt)
            stim_begin = int((t_init / h.dt))
            stim_end = stim_begin + stim_len

            while stim_end < len(time_steps):
                UDB_v[stim_begin:stim_end] = self.v_peak
                stim_begin = stim_end
                stim_end += stim_len

                UDB_v[stim_begin:stim_end] = self.v_held
                stim_begin += stim_len
                stim_end += stim_len

            return time_steps, UDB_v

        # vectors for data handling
        self.t_vec, self.v_vec = make_UDB()
        self.i_vec = []
        self.ipeak_vec = []
        self.peak_times = []
        self.norm_peak = []  # peak currents normalized to first current

    def clamp(self, v_cl):
        """Runs a trace and calculates currents.
        Args:
            v_cl (int): voltage to run
        """

        self.f3cl.amp[0] = v_cl
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.

        # parameters initialization
        stim_counter = 0
        dtsave = h.dt

        while h.t < h.tstop:  # runs a single trace, calculates current
            self.f3cl.amp[0] = self.v_vec[stim_counter]
            dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                0.5).i_cap  # clamping voltage in mA/cm2, for each dt

            self.i_vec.append(dens)  # trace to be plotted

            stim_counter += 1
            h.fadvance()

    def genUDB20(self):
        h.tstop = self.t_total
        self.clamp(self.v_vec[0])

    """ Currently not working: pulses 2-9 have the same peak current  
    def getPeakCurrs(self):
        for iter in range(self.num_repeats):
            peak_starts = int((self.t_init + (2 * self.t_peakdur * iter)) / h.dt)
            peak_ends = int(peak_starts + (2 * self.t_peakdur) / h.dt)

            self.ipeak_vec.append(max(self.i_vec[peak_starts: peak_ends - 1]))
            self.peak_times.append(self.t_vec[self.i_vec.index(self.ipeak_vec[iter])])
            self.norm_peak.append(self.ipeak_vec[iter] / self.ipeak_vec[0])
    """

    def plotUDB20_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title("UBD20: Time Voltage Relation")
        plt.plot(self.t_vec, self.v_vec, c='black')
        plt.yticks(np.arange(self.v_init, self.v_peak + 10, 10))
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/UDB20 Time Voltage relation'))

    def plotUDB20_TimeCurrentRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current $(pA)$')
        plt.title("UDB20: Current of Pulses")
        plt.plot(self.t_vec[1:], self.i_vec[1:], c='black')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/UDB20 Current of Pulses'))

    def plotAllUDB20(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotUDB20_TimeVRelation()
        self.plotUDB20_TimeCurrentRelation()


##################
# RFI dv tau
##################
class RFI_dv:
    def __init__(self, ntrials=30, recordTime=500,
                 soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.01,
                 min_inter=0.1, max_inter=5000, num_pts=50, cond_st_dur=1, res_pot=-120, dur=0.1,
                 vec_pts=np.linspace(-120, 0, num=13),
                 f3cl_dur0=50, f3cl_amp0=-120, f3cl_dur1=5, f3cl_amp1=0, f3cl_dur2=1,
                 f3cl_dur3=5, f3cl_amp3=0, f3cl_dur4=5, f3cl_amp4=-120):
        self.ntrials = ntrials
        self.recordTime = recordTime

        # one-compartment cell (soma)
        self.soma = h.Section(name='soma2')
        self.soma.diam = soma_diam  # micron
        self.soma.L = soma_L  # micron, so that area = 10000 micron2
        self.soma.nseg = soma_nseg  # dimensionless
        self.soma.cm = soma_cm  # uF/cm2
        self.soma.Ra = soma_Ra  # ohm-cm
        self.soma.insert(channel_name)  # insert mechanism
        self.soma.ena = soma_ena

        self.h = h
        self.h.celsius = h_celsius  # temperature in celsius
        self.v_init = v_init  # holding potential
        self.h.dt = h_dt  # ms - value of the fundamental integration time step, dt,
        # used by fadvance() in RecInactTau.
        # Increase value to speed up recInactTau().

        # clamping parameters for RecInactTau
        self.min_inter = min_inter  # pre-stimulus starting interval
        self.max_inter = max_inter  # pre-stimulus endinging interval
        self.num_pts = num_pts  # number of points in logaritmic scale
        self.cond_st_dur = cond_st_dur  # conditioning stimulus duration
        self.res_pot = res_pot  # resting potential
        self.dur = dur

        # vector containing 'num_pts' values equispaced between log10(min_inter) and log10(max_inter)
        # for RecInactTau
        # vec_pts = [1,1.5,3,5.6,10,30,56,100,150,300,560,1000,2930,5000]
        # vec_pts = np.logspace(np.log10(min_inter), np.log10(max_inter), num=num_pts)
        self.vec_pts = vec_pts
        self.L = len(vec_pts)

        # vectors for data handling RecInactTau
        self.rec_vec = h.Vector()
        self.time_vec = h.Vector()
        self.log_time_vec = h.Vector()
        self.t_vec = h.Vector()
        self.v_vec_t = h.Vector()
        self.i_vec_t = h.Vector()
        self.rec_inact_tau_vec = h.Vector()
        self.all_is = []

        # voltage clamp with "five" levels for RecInactTau
        self.f3cl = h.VClamp_plus(self.soma(0.5))
        self.f3cl.dur[0] = f3cl_dur0  # ms
        self.f3cl.amp[0] = f3cl_amp0  # mV
        self.f3cl.dur[1] = f3cl_dur1  # ms prev 1000
        self.f3cl.amp[1] = f3cl_amp1  # mV
        self.f3cl.dur[2] = f3cl_dur2  # ms
        self.f3cl.amp[2] = res_pot  # mV -120
        self.f3cl.dur[3] = f3cl_dur3  # ms
        self.f3cl.amp[3] = f3cl_amp3  # mV
        self.f3cl.dur[4] = f3cl_dur4  # ms
        self.f3cl.amp[4] = f3cl_amp4  # mV

    # clamping definition for RecInactTau
    def clampRecInact_dv_Tau(self, curr_amp):

        self.f3cl.amp[2] = curr_amp
        h.tstop = 50 + 5 + 1 + 5 + 5
        h.finitialize(self.v_init)

        # variables initialization
        pre_i1 = 0
        pre_i2 = 0
        dens = 0

        peak_curr1 = 0
        peak_curr2 = 0

        while (h.t < h.tstop):  # runs a single trace, calculates peak current

            dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                0.5).i_cap  # clamping current in mA/cm2, for each dt
            self.t_vec.append(h.t)
            self.v_vec_t.append(self.soma.v)
            self.i_vec_t.append(dens)

            if ((h.t > 5) and (h.t < 15)):  # evaluate the first peak
                if (pre_i1 < abs(dens)):
                    peak_curr1 = abs(dens)
                pre_i1 = abs(dens)

            if ((h.t > (5 + self.cond_st_dur + self.dur)) and (
                    h.t < (15 + self.cond_st_dur + self.dur))):  # evaluate the second peak

                if (pre_i2 < abs(dens)):
                    peak_curr2 = abs(dens)
                pre_i2 = abs(dens)

            h.fadvance()

        # updates the vectors at the end of the run
        self.time_vec.append(self.dur)
        self.log_time_vec.append(np.log10(self.dur))
        self.rec_vec.append(peak_curr2 / peak_curr1)

        # calc tau using RF and tstop
        # append values to vector
        RF_t = peak_curr2 / peak_curr1
        tau = -h.tstop / np.log(-RF_t + 1)
        self.rec_inact_tau_vec.append(tau)

    # start RecInact program and plot
    def plotRecInact_dv(self):

        # figure definition
        fig = plt.figure(figsize=(18, 16))
        ax5 = plt.subplot2grid((2, 4), (0, 0), colspan=2)

        fig.subplots_adjust(wspace=0.5)
        fig.subplots_adjust(hspace=0.5)

        # if time_vec is changed to see plot in not log time
        # then change xlim to (-150, 5 + cond_st_dur + max_inter + 20 + 5)
        # ax5.set_xlim(-1.1,1.1)
        # ax5.set_ylim(-0.1, 1.1)
        ax5.set_xlabel('Log(Time)')
        ax5.set_ylabel('Fractional recovery (P2/P1)')
        ax5.set_title('Log(Time)/Fractional recovery (P2/P1)')

        k = 0  # counter

        for amp in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)

            self.clampRecInact_dv_Tau(amp)
            k += 1

            # change log_time_vec to time_vec (ms) to see plot in not log time
            ln5 = ax5.scatter(self.time_vec, self.rec_vec, c="black")
            ax5.set_xscale('log')

        plt.show()

    # Generate RecInactTau
    # Returns rec_inact_tau_vec
    def genRecInactTau_dv(self):
        k = 0  # counter

        for amp in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)

            self.clampRecInact_dv_Tau(amp)
            k += 1
            aa = self.i_vec_t.to_python()
            self.all_is.append(aa[1:])
        return self.rec_inact_tau_vec, self.all_is

    def genRecInactTauCurve_dv(self):
        # figure definition
        recov = []
        times = []

        k = 0  # counter

        for dur in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)

            self.clampRecInact_dv_Tau(dur)
            k += 1

            recov.append(self.rec_vec.to_python()[0])

        return recov, self.vec_pts

    # Plot time/voltage relation (simulation) for RFI
    def plotRecInactProcedure_dv(self):

        # figure definition
        fig = plt.figure(figsize=(18, 6))
        ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2)

        fig.subplots_adjust(wspace=0.5)
        fig.subplots_adjust(hspace=0.5)

        ax0.set_xlim(-150, 5 + self.max_inter + 20 + 100)
        ax0.set_ylim(-121, 20)
        ax0.set_xlabel('Time $(ms)$')
        ax0.set_ylabel('Voltage $(mV)$')
        ax0.set_title('Time/Voltage Reltation')

        k = 0  # counter

        for dur in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)

            self.clampRecInact_dv_Tau(dur)
            k += 1

            ln5 = ax0.plot(self.t_vec, self.v_vec_t, c="black")

        plt.show()

    # Generate RFI data
    # Returns rec_vec
    def genRecInact_dv(self):
        k = 0  # counter

        rec_return = []
        for dur in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)

            self.clampRecInact_dv_Tau(dur)

            rec_return.append(self.rec_vec.to_python()[0])
            k += 1

        return rec_return, self.vec_pts

    def plotAllRecInact_with_ax(self, ax_list, cur_params):
        color = 'red' if cur_params == "variant_params" else 'black'
        y_offset = -0.2 if cur_params == "variant_params" else 0
        x_offset = 0

        # if time_vec is changed to see plot in not log time
        # then change xlim to (-150, 5 + cond_st_dur + max_inter + 20 + 5)
        # ax5.set_xlim(-1.1,1.1)
        # ax5.set_ylim(-0.1, 1.1)
        ax_list[0].set_xlabel('Log(Time)')
        ax_list[0].set_ylabel('Fractional recovery (P2/P1)')
        ax_list[0].set_title('Log(Time)/Fractional recovery (P2/P1)')
        k = 0  # counter
        for amp in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)

            self.clampRecInact_dv_Tau(amp)
            k += 1

            # change log_time_vec to time_vec (ms) to see plot in not log time
            ln5 = ax_list[0].scatter(self.time_vec, self.rec_vec, c=color)
            ax_list[0].set_xscale('log')

        ax_list[1].set_xlim(-150, 5 + self.max_inter + 20 + 100)
        ax_list[1].set_ylim(-121, 20)
        ax_list[1].set_xlabel('Time $(ms)$')
        ax_list[1].set_ylabel('Voltage $(mV)$')
        ax_list[1].set_title('Time/Voltage Reltation')
        k = 0  # counter
        for dur in self.vec_pts:
            # resizing the vectors
            self.t_vec.resize(0)
            self.i_vec_t.resize(0)
            self.v_vec_t.resize(0)
            self.rec_vec.resize(0)
            self.time_vec.resize(0)
            self.log_time_vec.resize(0)
            self.rec_inact_tau_vec.resize(0)
            self.clampRecInact_dv_Tau(dur)
            k += 1
            ln5 = ax_list[1].plot(self.t_vec, self.v_vec_t, c=color)


def fit_sigmoid(x, a, b):
    """
    Fit a sigmoid curve to the array of datapoints. 
    """
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def update_params(vc_params):
    nrn_h = activationNa12("geth")
    params = list(vc_params.keys())
    for p in params:
        nrn_h(p + '_na12mut =' + str(vc_params[p]))


def fit_exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def find_tau_inact(inact_i, ax=None):
    all_tau_sweeps = [-130 + i * 10 for i in
                      range(len(inact_i))]  # Need to change the constant every time. Need to fix.
    all_taus = []
    for i in range(len(inact_i)):
        raw_data = inact_i[i][1:]

        min_val, mindex = min((val, idx) for (idx, val) in enumerate(raw_data[:int(0.7 * len(raw_data))]))
        data = raw_data[mindex:mindex + 100]
        ts = [0.1 * i for i in range(len(data))]  # make x values which match sample times
        try:
            popt, pcov = optimize.curve_fit(fit_exp, ts, data)  # fit exponential curve
            perr = np.sqrt(np.diag(pcov))
            # print('in ' + str(all_tau_sweeps[i]) + ' the error was ' + str(perr))
            xs = np.linspace(ts[0], ts[len(ts) - 1], 1000)  # create uniform x values to graph curve
            ys = fit_exp(xs, *popt)  # get y values
            vmax = max(ys) - min(ys)  # get diff of max and min voltage
            vt = min(ys) + .37 * vmax  # get vmax*1/e
            tau = (np.log([(vt - popt[2]) / popt[0]]) / (-popt[1]))[0]  # find time at which curve = vt
        except:
            tau = 0
        if ax is not None:
            if all_tau_sweeps[i] == 0:
                ts = [0.1 * i for i in range(len(raw_data))]
                xs = xs + ts[mindex]
                ax.plot(ts, raw_data, color="red")
                tau_actual = tau + 0.1 * mindex  # adjust for slicing by adding time sliced off
                # ax.plot(xs, ys, color="blue")
                # plt.vlines(tau, min(ys)-.02, max(ys)+.02)
        # uncomment to plot
        #plt.vlines(tau, min(ys)-.02, max(ys)+.02)
        #plt.figure()
        #plt.plot(ts, data, color="black")
        #plt.plot(xs, ys, color="red")
        #plt.text(0, 0, i)
        #plt.show()

        all_taus.append(tau)

    tau_sweeps = []
    taus = []
    for i in range(len(all_tau_sweeps)):
        if all_tau_sweeps[i] >= -30:
            if all_tau_sweeps[i] == 0:
                tau0 = all_taus[i]
            tau_sweeps.append(all_tau_sweeps[i])
            taus.append(all_taus[i])
    return taus, tau_sweeps, tau0


def plot_act_inact_mut(new_params, wt_data):
    update_params(new_params)
    act, act_x, act_i = activationNa12("genActivation")
    act = list(act.to_python())
    act_x = list(act_x)
    inact, inact_x, inact_i = inactivationNa12("genInactivation")
    inact = list(inact.to_python())
    inact_x = list(inact_x)

    popt_act, pcov = optimize.curve_fit(fit_sigmoid, act_x, act, p0=[-.120, act[0]], maxfev=5000)
    popt_inact, pcov = optimize.curve_fit(fit_sigmoid, inact_x, inact, p0=[-.120, inact[0]], maxfev=5000)
    act_even_xs = np.linspace(act_x[0], act_x[len(act_x) - 1], 100)
    inact_even_xs = np.linspace(inact_x[0], inact_x[len(act_x) - 1], 100)
    act_curve = fit_sigmoid(act_even_xs, *popt_act)
    inact_curve = fit_sigmoid(inact_even_xs, *popt_inact)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(wt_data['act_x'], wt_data['act'], 'black')
    ax1.plot(wt_data['inact_x'], wt_data['inact'], 'black')
    ax1.plot(act_even_xs, act_curve, color='red')
    ax1.plot(inact_even_xs, inact_curve, color='red')
    ax1.set_xlabel('Volts[mV]')
    ax1.set_ylabel('Fraction Activated')
    # act_x = list(range(-120,40,10))
    # inact_x = list(range(-120,40,10))
    ax1.plot(act_x, act, 'o', color='red')
    ax1.plot(inact_x, inact, 'o', color='red')

    taus, tau_sweeps, tau0 = find_tau_inact(inact_i)
    popt, pcov = optimize.curve_fit(fit_exp, tau_sweeps, taus, maxfev=5000)
    tau_xs = np.linspace(tau_sweeps[0], tau_sweeps[len(tau_sweeps) - 1], 1000)
    tau_ys = fit_exp(tau_xs, *popt)
    ax2.plot(wt_data['tau_x'], wt_data['tau_y'], 'black')
    ax2.scatter(tau_sweeps, taus, color='red')
    ax2.plot(tau_xs, tau_ys, 'red')
    ax2.set_xlabel('Volts[mV]')
    ax2.set_ylabel('Inact tau[mS]')
    plt.show();
    fig.savefig('vclamp_mut.pdf')


def plot_act_inact_wt():
    plot_act = True
    act, act_x, act_i = activationNa12("genActivation")
    act = list(act.to_python())
    act_x = list(act_x)
    inact, inact_x, inact_i = inactivationNa12("genInactivation")
    inact = list(inact.to_python())
    inact_x = list(inact_x)
    popt_act, pcov = optimize.curve_fit(fit_sigmoid, act_x, act, p0=[-.120, act[0]], maxfev=5000)
    popt_inact, pcov = optimize.curve_fit(fit_sigmoid, inact_x, inact, p0=[-.120, inact[0]], maxfev=5000)
    act_even_xs = np.linspace(act_x[0], act_x[len(act_x) - 1], 100)
    inact_even_xs = np.linspace(inact_x[0], inact_x[len(act_x) - 1], 100)
    act_curve = fit_sigmoid(act_even_xs, *popt_act)
    inact_curve = fit_sigmoid(inact_even_xs, *popt_inact)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # taus, tau_sweeps, tau0 = find_tau_inact(inact_i) #this is when using inactivation protocol
    taus, tau_sweeps, tau0 = find_tau_inact(act_i)  # this is when using inactivation protocol

    popt, pcov = optimize.curve_fit(fit_exp, tau_sweeps, taus, maxfev=5000)
    tau_xs = np.linspace(tau_sweeps[0], tau_sweeps[len(tau_sweeps) - 1], 1000)
    tau_ys = fit_exp(tau_xs, *popt)

    ax1.plot(act_even_xs, act_curve, color='black')
    ax1.plot(inact_even_xs, inact_curve, color='black')
    # act_x = list(range(-120,40,10))
    # inact_x = list(range(-120,40,10))
    ax1.plot(act_x, act, 'o', color='black')
    ax1.plot(inact_x, inact, 'o', color='black')

    ax2.scatter(tau_sweeps, taus, color='black')
    ax2.plot(tau_xs, tau_ys, 'black')
    plt.show();
    ans = {}
    ans['act_x'] = act_even_xs
    ans['inact_x'] = inact_even_xs
    ans['act'] = act_curve
    ans['inact'] = inact_curve
    ans['tau_x'] = tau_xs
    ans['tau_y'] = tau_ys

    return ans


# wt_act_inact = plot_act_inact_wt()
# vclamp_params = {'tha_na12mut':-15,'qa_na12mut':7.2,'thinf_na12mut':-45,'qinf_na12mut':7}
# plot_act_inact_mut(vclamp_params,wt_act_inact)
# plot_act_inact_wt()
# recInactTauNa12("plotRecInact")


#######################
# MAIN
#######################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulated data.')
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    if args.function == 1:
        genAct = Activation(channel_name='na12mut8st')
        genAct.genActivation()
        genAct.plotAllActivation()

    elif args.function == 2:
        genInact = Inactivation(channel_name='na12mut8st')
        genInact.genInactivation()
        genInact.plotAllInactivation()


    elif args.function == 3:
        genRFI = RFI(channel_name='na12mut8st')
        genRFI.genRecInactTau()
        genRFI.plotAllRFI()

    elif args.function == 4:
        genRamp = Ramp()
        genRamp.genRamp()
        genRamp.plotAllRamp()

    elif args.function == 5:
        genRFIdv = RFI_dv()
        genRFIdv.genRecInact_dv()
        genRFIdv.genRecInactTau_dv()
        genRFIdv.genRecInactTauCurve_dv()
        genRFIdv.plotRecInact_dv()
        genRFIdv.plotRecInactProcedure_dv()

    elif args.function == 6:
        genUDB20 = UDB20()
        genUDB20.genUDB20()
        genUDB20.plotAllUDB20()

    elif args.function == 7:
        # run all
        genAct = Activation()
        genAct.genActivation()
        genAct.plotAllActivation()

        genInact = Inactivation()
        genInact.genInactivation()
        genInact.plotAllInactivation()

        genRFI = RFI()
        genRFI.genRecInactTau()
        genRFI.plotAllRFI()

        genRamp = Ramp()
        genRamp.genRamp()
        genRamp.plotAllRamp()

        genUDB20 = UDB20()
        genUDB20.genUDB20()
        genUDB20.plotAllUDB20()

    elif args.function == 8:
        # run all with saving ax
        # change channel name accordingly for HH models
        # na12 (na12.mod)
        # na16 (na16.mod)
        genAct = Activation(channel_name='na12')
        genAct.genActivation()
        genAct.plotAllActivation_with_ax(fig_title="Activation HH vs HMM", color='black',
                                         saveAsFileName="Plots_Folder/Act HHvHMM", loadFileName=None,
                                         saveAsPNGFileName="Plots_Folder/Act HHvHMM")

        genInact = Inactivation(channel_name='na12')
        genInact.genInactivation()
        genInact.plotAllInactivation_with_ax(fig_title="Inactivation HH vs HMM", color='black',
                                             saveAsFileName="Plots_Folder/Inact HHvHMM", loadFileName=None,
                                             saveAsPNGFileName="Plots_Folder/Inact HHvHMM")

        genRFI = RFI(channel_name='na12')
        genRFI.genRecInactTau()
        genRFI.plotAllRFI_with_ax(fig_title="RFI HH vs HMM", color='black',
                                  saveAsFileName="Plots_Folder/RFI HHvHMM", loadFileName=None,
                                  saveAsPNGFileName="Plots_Folder/RFI HHvHMM")
    elif args.function == 9:
        # run all with saving ax
        # change channel name accordingly for HMM models
        # na (na8st.mod) (corresponding na12)
        # nax (na8xst.mod) (corresponding na16)
        genAct = Activation(channel_name='na')
        genAct.genActivation()
        genAct.plotAllActivation_with_ax(fig_title="Activation HH vs HMM", color='red',
                                         saveAsFileName="Plots_Folder/Act HHvHMM",
                                         loadFileName="Plots_Folder/Act HHvHMM",
                                         saveAsPNGFileName="Plots_Folder/Act HHvHMM")

        genInact = Inactivation(channel_name='na')
        genInact.genInactivation()
        genInact.plotAllInactivation_with_ax(fig_title="Inactivation HH vs HMM", color='red',
                                             saveAsFileName="Plots_Folder/Inact HHvHMM",
                                             loadFileName="Plots_Folder/Inact HHvHMM",
                                             saveAsPNGFileName="Plots_Folder/Inact HHvHMM")

        genRFI = RFI(channel_name='na')
        genRFI.genRecInactTau()
        genRFI.plotAllRFI_with_ax(fig_title="RFI HH vs HMM", color='red',
                                  saveAsFileName="Plots_Folder/RFI HHvHMM",
                                  loadFileName="Plots_Folder/RFI HHvHMM",
                                  saveAsPNGFileName="Plots_Folder/RFI HHvHMM")
