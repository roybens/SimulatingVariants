""" VClamp SCN2A Variants (HMM)
 Generates simulated data for voltage-gated channel
 Hidden Markov Model
 Modified from Emilio Andreozzi "Phenomenological models of NaV1.5.
    # A side by side, procedural, hands-on comparison between
    # Hodgkin-Huxley and kinetic formalisms." 2019
 Contributors: Emily Nguyen UC Berkeley, Roy Ben-Shalom UCSF
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
from state_variables import finding_state_variables
import pickle
import curve_fitting as cf

from generate_simulation import *

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

    def clamp_at_volt(self, v_cl):
        self.t_vec = []
        self.v_vec_t = []
        self.i_vec = []
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
        peak,ttp = self.find_ipeaks()
        self.ipeak_vec.append(peak)
        self.ttp_vec.append(ttp)

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
            curr_index = np.argmax(self.i_vec)
        else:
            curr_tr = curr_min
            curr_index = np.argmin(self.i_vec)
        return curr_tr, self.t_vec[curr_index]
    
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

<<<<<<< HEAD

        
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
        time_padding = 5  # ms
        h.tstop = time_padding + self.dur + time_padding  # time stop
        self.ipeak_vec = []
        self.all_is = []
        self.all_v_vec_t = []
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

    def get_Tau_0mV(self,upper = 700):
        def fit_expon(x, a, b, c):
            return a + b * np.exp(-1 * c * x)
        
        def one_phase(x, y0, plateau, k):
            return y0 + (plateau - y0) * (1 - np.exp(-k * x))
        self.clamp_at_volt(0)
        starting_index = list(self.i_vec).index(self.find_ipeaks_with_index()[1])
        
        t_vecc = self.t_vec[starting_index:upper]
        i_vecc = self.i_vec[starting_index:upper]
        try:
            popt, pcov = optimize.curve_fit(fit_expon,t_vecc,i_vecc, method = 'dogbox')
            fit = 'exp'
            tau = 1/popt[2]
            fitted_i = fit_expon(self.t_vec[starting_index:upper],popt[0],popt[1],popt[2])
        except:
            popt, pcov = optimize.curve_fit(one_phase,t_vecc,i_vecc, method = 'dogbox')
            fit = 'one_phase'
            tau = 1/popt[2]
            fitted_i = one_phase(act.t_vec[starting_index:upper],popt[0],popt[1],popt[2])
        return tau
    def plotActivation_VGnorm(self):
        """
        Saves activation plot as PGN file.
        """
        plt.figure()
        plt.xlabel('Voltage $(mV)$')
        plt.ylabel('Normalized conductance')
        plt.title('Activation: Voltage/Normalized conductance')
        plt.plot(self.v_vec, self.gnorm_vec, 'o', c='black')
        gv_slope, v_half, top, bottom = cf.calc_act_obj(self)
        formatted_gv_slope = np.round(gv_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5, f'Slope: {formatted_gv_slope}')
        plt.text(-10, 0.3, f'V50: {formatted_v_half}')
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c='red')
        # save as PGN file
        plt.savefig(
            os.path.join(os.path.split(__file__)[0],
                         'Plots_Folder/HMM_Activation Voltage-Normalized Conductance Relation'))

    def plotActivation_IVCurve(self):
        plt.figure()
        plt.xlabel('Voltage $(mV)$')
        plt.ylabel('Peak Current $(pA)$')
        plt.title("Activation: IV Curve")
        plt.plot(self.v_vec, self.ipeak_vec, 'o', c='black')
        plt.text(-110, -0.05, 'Vrev at ' + str(round(self.vrev, 1)) + ' mV', fontsize=10, c='blue')
        formatted_peak_i = np.round(min(self.ipeak_vec), decimals=2)
        plt.text(-110, -0.1, f'Peak Current from IV: {formatted_peak_i} pA', fontsize=10, c='blue')  # pico Amps
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Activation IV Curve"))

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
        # save as PGN file
        plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/HMM_Activation Time Voltage Relation'))

    def plotActivation_TCurrDensityRelation(self,xlim = None):
        plt.figure()
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current density $(mA/cm^2)$')
        plt.title('Activation Time/Current density relation')
        curr = np.array(self.all_is)
        [plt.plot(self.t_vec[1:], curr[i], c='black') for i in np.arange(len(curr))]
        if xlim is not None:
            for i in np.arange(len(curr)):
                plt.plot(self.t_vec[1:], curr[i], c='black')
                plt.xlim(xlim)

        # save as PGN file
        plt.savefig(
            os.path.join(os.path.split(__file__)[0], "Plots_Folder/HMM_Activation Time Current Density Relation"))
        
    def plotActivation_TCurrDensityRelation_plt(self,plt,color):
        curr = np.array(self.all_is)
        mask = np.where(np.logical_or(self.v_vec == 0, self.v_vec == 10))
        [plt.plot(self.t_vec[190:300], curr[i][190:300], c=color) for i in np.arange(len(curr))[mask]]
        
    
    def plotActivation_VGnorm_plt(self,plt,color):
        """
        Saves activation plot as PGN file.
        """
        
        diff = 0
        if color == 'red':
            diff = 0.5 
        
        
        plt.plot(self.v_vec, self.gnorm_vec, 'o', c=color)
        gv_slope, v_half, top, bottom = cf.calc_act_obj(self)
        #gv_slope, v_half, top, bottom = cf.calc_act_obj(self.channel_name, True)
        formatted_gv_slope = np.round(gv_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        plt.text(-10, 0.5 + diff, f'Slope: {formatted_gv_slope}', c = color)
        plt.text(-10, 0.3 + diff, f'V50: {formatted_v_half}', c = color)
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
        plt.plot(x_values_v, curve, c=color)
        return (formatted_v_half, formatted_gv_slope)

    def plotAllActivation(self):
        """
        Saves all plots to CWD/Plots_Folder.
        """
        self.plotActivation_VGnorm()
        self.plotActivation_IVCurve()
        self.plotActivation_TimeVRelation()
        self.plotActivation_TCurrDensityRelation()

    def plotAllActivation_with_ax(self, fig_title,
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
        x_offset = 0

        # upper left
        ax[0, 0].set_xlabel('Voltage $(mV)$')
        ax[0, 0].set_ylabel('Normalized conductance')
        ax[0, 0].set_title('Activation: Voltage/Normalized conductance')
        ax[0, 0].plot(self.v_vec, self.gnorm_vec, 'o', c=color)
        gv_slope, v_half, top, bottom = cf.calc_act_obj()
        formatted_gv_slope = np.round(gv_slope, decimals=2)
        formatted_v_half = np.round(v_half, decimals=2)
        ax[0, 0].text(-10 + x_offset, 0.5 + y_offset, f'Slope: {formatted_gv_slope}', color=color)
        ax[0, 0].text(-10 + x_offset, 0.4 + y_offset, f'V50: {formatted_v_half}', color=color)
        x_values_v = np.arange(self.st_cl, self.end_cl, 1)
        curve = cf.boltzmann(x_values_v, gv_slope, v_half, top, bottom)
        ax[0, 0].plot(x_values_v, curve, c=color, label='HHM')
        ax[0, 0].legend(loc='upper left')

        # lower left
        ax[1, 0].set_xlabel('Voltage $(mV)$')
        ax[1, 0].set_ylabel('Peak Current')
        ax[1, 0].set_title("Activation: IV Curve")
        ax[1, 0].plot(self.v_vec, self.ipeak_vec, 'o', c=color)
        ax[1, 0].text(-110 + x_offset, -0.05 + y_offset, 'Vrev at ' + str(round(self.vrev, 1)) + 'mV', c=color)
        formatted_peak_i = np.round(min(self.ipeak_vec), decimals=2)
        ax[1, 0].text(-110 + x_offset, -0.1 + y_offset, f'Peak Current from IV: {formatted_peak_i} mV', c=color)

        # upper right
        ax[0, 1].set_xlabel('Time $(ms)$')
        ax[0, 1].set_ylabel('Voltage $(mV)$')
        ax[0, 1].set_title('Activation Time/Voltage relation')
        [ax[0, 1].plot(self.t_vec, self.all_v_vec_t[i], c=color) for i in np.arange(self.L)]

        # lower right
        ax[1, 1].set_xlabel('Time $(ms)$')
        ax[1, 1].set_ylabel('Current density $(mA/cm^2)$')
        ax[1, 1].set_title('Activation Time/Current density relation')
        curr = np.array(self.all_is)
        [ax[1, 1].plot(self.t_vec[1:], curr[i], c=color) for i in np.arange(len(curr))]

        if saveAsFileName:
            with open(saveAsFileName, 'wb') as fid:
                pickle.dump(ax, fid)
        if saveAsPNGFileName:
            plt.savefig(
                os.path.join(os.path.split(__file__)[0], saveAsPNGFileName))


=======
>>>>>>> 7e373254e719c31b183953f5e0f4d1ddfb21112e
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


    def find_tau0_inact(self, raw_data):
        # take peak curr and onwards
        min_val, mindex = min((val, idx) for (idx, val) in enumerate(raw_data[:int(0.7 * len(raw_data))]))
        padding = 15  # after peak
        data = raw_data[mindex:mindex + padding]
        ts = [0.1 * i for i in range(len(data))]  # make x values which match sample times

        # calc tau and fit exp
        popt, pcov = optimize.curve_fit(fit_exp, ts, data)  # fit exponential curve
        perr = np.sqrt(np.diag(pcov))
        # print('in ' + str(all_tau_sweeps[i]) + ' the error was ' + str(perr))
        xs = np.linspace(ts[0], ts[len(ts) - 1], 1000)  # create uniform x values to graph curve
        ys = fit_exp(xs, *popt)  # get y values
        vmax = max(ys) - min(ys)  # get diff of max and min voltage
        vt = min(ys) + .37 * vmax  # get vmax*1/e
        #tau = (np.log([(vt - popt[2]) / popt[0]]) / (-popt[1]))[0]  # find time at which curve = vt
        #Roy said tau should just be the parameter b from fit_exp
        tau = popt[1]
        return ts, data, xs, ys, tau
    
    
##################
# Recovery from Inactivation (RFI)
# &  RFI Tau
##################
class RFI(RFI_general):
    def placeholder(self):
        return None


##################
# Ramp Protocol
##################
class Ramp(Ramp_general):



    def persistentCurrent(self):
        """ Calculates persistent current (avg current of last 100 ms at 0 mV)
        Normalized by peak from IV (same number as areaUnderCurve).
        """
        persistent = self.i_vec[self.t_start_persist:self.t_end_persist]
        act = Activation()
        act.genActivation()
        IVPeak = min(act.ipeak_vec)
        return (sum(persistent) / len(persistent)) / IVPeak


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
