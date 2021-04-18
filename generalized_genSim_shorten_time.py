"""
Written by Emily Nguyen, UC Berkeley
           Chastin Chung, UC Berkeley
           Isabella Boyle, UC Berkeley
           Roy Ben-Shalom, UCSF
    
Generates simulated data.
Modified from Emilio Andreozzi "Phenomenological models of NaV1.5.
    A side by side, procedural, hands-on comparison between Hodgkin-Huxley and kinetic formalisms." 2019
"""

from neuron import h, gui
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import optimize, stats
import argparse
import os

import curve_fitting as cf
#from sys import api_version
#from test.pythoninfo import collect_platform


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
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.025, ntrials=range(30),
                 dur=100, step=10, st_cl=-120, end_cl=40, v_cl=-120,
                 f3cl_dur0=5, f3cl_amp0=-120, f3cl_dur2=5, f3cl_amp2=-120,
                 ):

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

                self.t_vec.append(h.t)
                self.v_vec_t.append(self.soma.v)
                self.i_vec.append(dens)

                if (h.t > 5) and (h.t <= 10):  # evaluate the peak
                    if abs(dens) > abs(pre_i):
                        curr_tr = dens  # updates the peak current

                h.fadvance()
                pre_i = dens

        # updates the vectors at the end of the run
        self.ipeak_vec.append(curr_tr)

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
            for v_cl in np.arange(self.st_cl, self.end_cl, self.step):  # self.vec # TODO change stim 1:?
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
        gv_slope, v_half, top, bottom = cf.Curve_Fitter().calc_act_obj()
        formatted_gv_slope = np.round(gv_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5, f'Slope: {formatted_gv_slope}')
        plt.text(-10, 0.3, f'V50: {formatted_v_half}')
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c='red')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Activation Voltage-Normalized Conductance Relation'))

    def plotActivation_IVCurve(self):
        plt.figure()
        plt.xlabel('Voltage $(mV)$')
        plt.ylabel('Peak Current')
        plt.title("Activation: IV Curve")
        plt.plot(self.v_vec, self.ipeak_vec, 'o', c='black')
        plt.text(-110, -0.05, 'Vrev at ' + str(round(self.vrev, 1)) + 'mV', fontsize=10, c='blue')
        formatted_peak_i = np.round(min(self.ipeak_vec), decimals=2)
        plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} mV', fontsize=10, c='blue')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Activation IV Curve"))

    def plotActivation_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('Activation Time/Voltage relation')
        [plt.plot(self.t_vec, self.all_v_vec_t[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Activation Time Voltage Relation'))

    def plotActivation_TCurrDensityRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('Activation Time/Current density relation')
        # TODO extend x-axis by 1.00
        mask = np.where(self.v_vec < 20)  # current densities up to 20 mV
        curr = np.array(self.all_is)[mask]
        t = np.array(self.t_vec[1:])[mask]
        [plt.plot(t[i], curr[i], c='black') for i in mask]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Activation Time Current Density Relation"))

    def plotAllActivation(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotActivation_VGnorm()
        self.plotActivation_IVCurve()
        self.plotActivation_TimeVRelation()
        self.plotActivation_TCurrDensityRelation()

##################
# Inactivation
##################
class Inactivation:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.025, ntrials=range(30),
                 dur=500, step=10, st_cl=-120, end_cl=40, v_cl=-120,
                 f3cl_dur0=40, f3cl_amp0=-120, f3cl_dur2=20, f3cl_amp2=-10):

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
        t_peak = 0
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

                if (h.t >= 540) and (h.t <= 542):  # evaluate the peak
                    if abs(dens) > abs(peak_curr):
                        peak_curr = dens
                        t_peak = h.t

                h.fadvance()

        # updates the vectors at the end of the run
        self.ipeak_vec.append(peak_curr)

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
        ssi_slope, v_half, top, bottom = cf.Curve_Fitter().calc_inact_obj()
        formatted_ssi_slope = np.round(ssi_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5, f'Slope: {formatted_ssi_slope}')
        plt.text(-10, 0.3, f'V50: {formatted_v_half}')
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, ssi_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c='red')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Inactivation Voltage Normalized Current Relation'))

    def plotInactivation_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('Inactivation Time/Voltage relation')
        [plt.plot(self.t_vec, self.all_v_vec_t[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Inactivation Time Voltage Relation'))

    def plotInactivation_TCurrDensityRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('Inactivation Time/Current density relation')
        [plt.plot(self.t_vec[1:], self.all_is[i], c='black') for i in np.arange(self.L)]
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Inactivation Time Current Density Relation"))

    def plotAllInactivation(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotInactivation_VInormRelation()
        self.plotInactivation_TimeVRelation()
        self.plotInactivation_TCurrDensityRelation()

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
        self.time_vec = [] # same as time in vec_pts
        self.log_time_vec = [] # same as time in vec_ptsm but logged
        self.t_vec = []  # vector for time steps (h.dt)
        self.v_vec_t = []  # vector for voltage as function of time
        self.i_vec_t = []  # vector for current
        self.rec_inact_tau_vec = []  # RFI taus
        self.all_is = []  # all currents
        self.all_v_vec_t = []  # all voltages
        self.all_t_vec = [] # all h.t

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
        self.rec_vec.append(peak_curr2 / peak_curr1)

        # calc tau using RF and tstop
        # append values to vector
        RF_t = peak_curr2 / peak_curr1
        tau = -h.tstop / np.log(-RF_t + 1)
        self.rec_inact_tau_vec.append(tau)

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
        # TODO plot tau fast..etc
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
        y0, plateau, percent_fast, k_fast, k_slow, tau0 = cf.Curve_Fitter().calc_recov_obj()
        formatted_tauSlow = np.round(1 / k_slow, decimals=2)
        formatted_tauFast = np.round(1 / k_fast, decimals=2)
        formatted_percentFast = np.round(percent_fast, decimals=4)
        # TODO move text to RHS
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
        #self.plotRFI_LogVInormRelation()
        #self.plotRFI_TimeVRelation()
        #elf.plotRFI_TCurrDensityRelation()


##################
# Ramp Protocol
##################
class Ramp:
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, t_init = 30,
                 v_first_step = -60, t_first_step = 30, v_ramp_end = 0, t_ramp = 300, t_plateau = 100, 
                 v_last_step = -120, t_last_step = 30 ,h_dt=0.025):
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
        def make_ramp():
            time_steps_arr = np.array([t_init,t_first_step,t_ramp,t_plateau,t_last_step])
            time_steps_arr = (time_steps_arr/h_dt).astype(int)
            time_steps_arr = np.cumsum(time_steps_arr)
            ntimesteps = time_steps_arr[-1]
            ramp_v = np.zeros(ntimesteps)
            ramp_v[0:time_steps_arr[0]] = v_init
            ramp_v[time_steps_arr[0]:time_steps_arr[1]] = v_first_step
            ramp_v[time_steps_arr[1]:time_steps_arr[2]] = np.linspace(v_first_step,v_ramp_end,time_steps_arr[2]-time_steps_arr[1])
            ramp_v[time_steps_arr[2]:time_steps_arr[3]] = v_ramp_end
            ramp_v[time_steps_arr[3]:time_steps_arr[4]] = v_last_step
            return ramp_v
        
        self.ntrials = 1  #
        h.celsius = h_celsius  # temperature in celsius
        self.stim_ramp = make_ramp()  # the voltage of the whole protocol
        h.dt = h_dt  # ms - value of the fundamental integration time step, dt, used by fadvance().
        self.v_init = v_init  # holding potential
        self.t_start_persist = int((t_init + t_first_step + t_ramp) / h_dt) #time that plateau starts
        self.t_end_persist = int((t_init + t_first_step + t_ramp + t_plateau) / h_dt) #time that plateau ends
        self.t_total = t_init + t_first_step + t_ramp + t_plateau + t_last_step

        # a two-electrodes voltage clamp
        self.f3cl = h.VClamp(self.soma(0.5))
        self.f3cl.dur[0] = 1e9
        self.f3cl.amp[0] = self.stim_ramp[0]
        
        # vectors for data handling
        self.t_vec = np.ones(len(self.stim_ramp))*h_dt
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
        area = trapz(self.i_vec, x=self.v_vec_t)  # find area
        act = Activation()
        act.genActivation()
        print(self.i_vec)
        area = area / min(act.ipeak_vec)  # normalize to peak currents from activation
        return area
    
    def persistentCurrent(self):
        """ Calculates persistent current (avg current of last 100 ms at 0 mV)
        """
        persistent = self.i_vec[self.t_start_persist:self.t_end_persist]
        return sum(persistent)/len(persistent)
    
    def plotRamp_TimeVRelation(self):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(mV)$')
        plt.title('Ramp Time/Voltage relation')
        plt.plot(self.t_vec, self.v_vec, color='black')
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/Ramp Time Voltage relation'))
    
    def plotRamp_TimeCurrentRelation(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        f.add_subplot(111, frameon=False) #for shared axes labels big title
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.title("Ramp: Time Current Density Relation", x=0.4, y=1.1)
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current', labelpad= 25)
        
        # starting + first step + ramp section
        ax1.set_title("Ramp")
        ax1.plot(self.t_vec[1:self.t_start_persist], self.i_vec[1:self.t_start_persist], 'o', c='black', markersize = 0.1)
        
        # persistent current + last step section
        ax2.set_title("Persistent Current")
        ax2.plot(self.t_vec[self.t_start_persist:], self.i_vec[self.t_start_persist:], 'o', c='black', markersize = 0.1)
        
        # save as PGN file
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/Ramp Time Current Density Relation"))

    def plotAllRamp(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotRamp_TimeVRelation()
        self.plotRamp_TimeCurrentRelation()

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

            dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(0.5).i_cap  # clamping current in mA/cm2, for each dt
            self.t_vec.append(h.t)
            self.v_vec_t.append(self.soma.v)
            self.i_vec_t.append(dens)

            if ((h.t > 5) and (h.t < 15)):  # evaluate the first peak
                if (pre_i1 < abs(dens)):
                    peak_curr1 = abs(dens)
                pre_i1 = abs(dens)

            if ((h.t > (5 + self.cond_st_dur + self.dur)) and (h.t < (15 + self.cond_st_dur + self.dur))):  # evaluate the second peak

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
                # tau_actual = tau+0.1*mindex #adjust for slicing by adding time sliced off

                ax.plot(xs, ys, color="blue")
                # plt.vlines(tau, min(ys)-.02, max(ys)+.02)
        # 
        # plt.plot(ts, data, color="red")
        # plt.plot(xs, ys, color="orange")
        # plt.vlines(tau, min(ys)-.02, max(ys)+.02)
        # plt.show()
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
        genAct = Activation()
        genAct.genActivation()
        genAct.plotAllActivation()

    elif args.function == 2:
        genInact = Inactivation()
        genInact.genInactivation()
        genInact.plotAllInactivation()

    elif args.function == 3:
        genRFI = RFI()
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
        pass