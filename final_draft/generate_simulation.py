"""
Written by Emily Nguyen, UC Berkeley
           Chastin Chung, UC Berkeley
           Isabella Boyle, UC Berkeley
           Jinan Jiang, UC Berkeley
           Roy Ben-Shalom, UCSF
           
    
Generates simulated data.
Modified from Emilio Andreozzi "Phenomenological models of NaV1.5.
    A side by side, procedural, hands-on comparison between Hodgkin-Huxley and kinetic formalisms." 2019
"""

from neuron import h, gui
import numpy as np
from numpy import trapz
from scipy import optimize, stats

class General_protocol:
    def __init__(self, channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena):
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
        self.f3cl = h.VClamp(self.soma(0.5))
        self.v_init = -65
        self.t_vec = []  # vector for time steps (h.dt)
        self.v_vec = []
        self.v_vec_t = []  # vector for voltage as function of time
        self.i_vec = []  # vector for current
    def get_h(self):
        return self.h
    def update_clamp_time_step(self):
        dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                    0.5).i_cap  # clamping current in mA/cm2, for each dt
        # append data
        self.t_vec.append(h.t)
        self.v_vec_t.append(self.soma.v)
        self.i_vec.append(dens)

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

    def find_ipeaks(self,ranges = [4,10]):
        """
        Evaluate the peak and updates the peak current.
        Returns peak current.
        Finds positive and negative peaks.
        """
        self.i_vec = np.array(self.i_vec)
        self.t_vec = np.array(self.t_vec)
        mask = np.where(np.logical_and(self.t_vec >= ranges[0], self.t_vec <= ranges[1]))
        i_slice = self.i_vec[mask]
        curr_max = np.max(i_slice)
        curr_min = np.min(i_slice)
        if np.abs(curr_max) > np.abs(curr_min):
            curr_tr = curr_max
        else:
            curr_tr = curr_min
        curr_index = np.where(self.i_vec == curr_tr)[0][0]
        return curr_tr, self.t_vec[curr_index],curr_index

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


class Activation_general(General_protocol):
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.025,
                 dur=20, step=5, st_cl=-120, end_cl=40, v_cl=-120,
                 f3cl_dur0=5, f3cl_amp0=-120, f3cl_dur2=5, f3cl_amp2=-120,
                 ):
        super().__init__(channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena)

        
        # clamping parameters
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
        self.ttp_vec = [] # vector for time to peak
        self.gnorm_vec = []  # vector for normalized conductance
        self.all_is = []  # all currents
        self.all_v_vec_t = []

        self.L = len(self.v_vec)

        # conductance attributes for plotting
        self.vrev = 0
        self.v_half = 0
        self.s = 0
        
        self.channel_name = channel_name

    def clamp(self, v_cl):
        """ Runs a trace and calculates peak currents.
        Args:
            v_cl (int): voltage to run
        """
        time_padding = 5  # ms
        h.tstop = time_padding + self.dur + time_padding  # time stop
        self.t_vec = []
        self.v_vec_t = []
        self.i_vec = []
        curr_tr = 0  # initialization of peak current
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.
        pre_i = 0  # initialization of variables used to commute the peak current
        dens = 0
        self.f3cl.amp[1] = v_cl  # mV
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
        peak, ttp,_ = self.find_ipeaks()
        self.ipeak_vec.append(peak)
        self.ttp_vec.append(ttp)


        
    def findG(self, v_vec, ipeak_vec):
        #same
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
        #same
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
            v_cls = np.arange(self.st_cl, self.end_cl, self.step)
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

    def get_Tau_0mV(self, upper=700):
        def fit_expon(x, a, b, c):
            return a + b * np.exp(-1 * c * x)

        def one_phase(x, y0, plateau, k):
            return y0 + (plateau - y0) * (1 - np.exp(-k * x))

        self.clamp(0)
        curr_peak, peak_time, peak_index = self.find_ipeaks()
        starting_index = peak_index
        print(f' starting index is: {starting_index}')
        t_vecc = self.t_vec[starting_index:upper]
        i_vecc = self.i_vec[starting_index:upper]
        try:
            popt, pcov = optimize.curve_fit(fit_expon, t_vecc, i_vecc, method='dogbox')
            fit = 'exp'
            tau = 1 / popt[2]
            fitted_i = fit_expon(self.t_vec[starting_index:upper], popt[0], popt[1], popt[2])
        except:
            popt, pcov = optimize.curve_fit(one_phase, t_vecc, i_vecc, method='dogbox')
            fit = 'one_phase'
            tau = 1 / popt[2]
            fitted_i = one_phase(self.t_vec[starting_index:upper], popt[0], popt[1], popt[2])
        return tau

    def find_peak_amp(self, ranges=None):
        if not self.ipeak_vec:
            #print('regen activation in peak_amp')
            self.genActivation()
        if ranges is None:
            return self.ipeak_vec
        else:
            return self.ipeak_vec[ranges[0]:ranges[1]]

    def find_time_to_peak(self, ranges=None):
        if not self.ttp_vec:
            #print('regen activation in ttp')
            self.genActivation()
        if ranges is None:
            return self.ttp_vec
        else:
            return self.ttp_vec[ranges[0]:ranges[1]]



    
class Inactivation_general(General_protocol):
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.025,
                 dur=500, step=5, st_cl=-120, end_cl=40, v_cl=-120,
                 f3cl_dur0=40, f3cl_amp0=-120, f3cl_dur2=20, f3cl_amp2=-10):

        #same
        super().__init__(channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena)

        # clamping parameters
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
        self.st_step_time = self.f3cl.dur[0] + self.f3cl.dur[1]
        self.end_step_time = self.f3cl.dur[0] + self.f3cl.dur[1] + self.f3cl.dur[2]
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
        while h.t < h.tstop:  # runs a single trace, calculates peak current
            if h.t > self.st_step_time-10*dtsave:
                h.dt = dtsave
            else:
                h.dt = 1
            dens = self.f3cl.i / self.soma(0.5).area() * 100.0 - self.soma(
                0.5).i_cap  # clamping current in mA/cm2, for each dt
            self.t_vec.append(h.t)  # code for store the current
            self.v_vec_t.append(self.soma.v)  # trace to be plotted
            self.i_vec.append(dens)  # trace to be plotted

            if (h.t >= self.st_step_time) and (h.t <= self.st_step_time + self.f3cl.dur[2]):  # evaluate the peak
                if abs(dens) > abs(peak_curr):
                    peak_curr = dens
                    t_peak = h.t

            h.fadvance()

        # updates the vectors at the end of the run
        self.ipeak_vec.append(peak_curr)
        
    def genInactivation(self):
        #same
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

        
class RFI_general(General_protocol):
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-90, h_dt=0.1,
                 min_inter=0.1, max_inter=5000, num_pts=50, cond_st_dur=1000, res_pot=-90, dur=0.1,
                 vec_pts=[1, 1.5, 3, 5.6, 10, 30, 56, 100, 150, 300, 560, 1000, 2930, 5000],
                 f3cl_dur0=5, f3cl_amp0=-90, f3cl_amp1=0, f3cl_dur3=20, f3cl_amp3=0, f3cl_dur4=5, f3cl_amp4=-90):

        #same
        super().__init__(channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena)

        # clamping parameters
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
        #same
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

            self.update_clamp_time_step()

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

class Ramp_general(General_protocol):
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, t_init=30,
                 v_first_step=-60, t_first_step=30, v_ramp_end=0, t_ramp=300, t_plateau=100,
                 v_last_step=-120, t_last_step=30, h_dt=0.025):
        #same
        super().__init__(channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena)
        
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
        #same
        self.f3cl.amp[0] = v_cl
        h.finitialize(self.v_init)  # calling the INITIAL block of the mechanism inserted in the section.
        # parameters initialization
        stim_counter = 0

        dtsave = h.dt
        while round(h.t, 3) < h.tstop:  # runs a single trace, calculates current
            self.f3cl.amp[0] = self.stim_ramp[stim_counter]
            self.update_clamp_time_step()

            stim_counter += 1
            h.fadvance()

    def genRamp(self):
        #same
        h.tstop = self.t_total
        self.clamp(self.v_vec[0])

    def areaUnderCurve(self):
        #same
        """ Calculates and returns normalized area (to activation IV) under IV curve of Ramp
        """
        maskStart, maskEnd = self.time_steps_arr[1], self.time_steps_arr[2]  # selects ramp (incline) portion only
        i_vec_ramp = self.i_vec[maskStart:maskEnd]
        v_vec_t_ramp = self.v_vec_t[maskStart:maskEnd]
        # plt.plot(self.t_vec[maskStart:maskEnd], self.v_vec[maskStart:maskEnd], color= 'b') # uncomment to view area taken
        area = trapz(i_vec_ramp, x=v_vec_t_ramp)  # find area
        act = Activation(channel_name='na12')
        act.genActivation()
        area = area / min(act.ipeak_vec)  # normalize to peak currents from activation
        return area

class UDB20_general(General_protocol):
    def __init__(self, soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, v_held=-70,
                 v_peak=-10, t_peakdur=100, t_init=200, num_repeats=9, h_dt=0.025):
        #same
        super().__init__(channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena)

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
        #same
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
            self.update_clamp_time_step()

            stim_counter += 1
            h.fadvance()

    def genUDB20(self):
        #same
        h.tstop = self.t_total
        self.clamp(self.v_vec[0])
        
        
class RFI_dv_general(General_protocol):
    def __init__(self, recordTime=500,
                 soma_diam=50, soma_L=63.66198, soma_nseg=1, soma_cm=1, soma_Ra=70,
                 channel_name='na12mut', soma_ena=55, h_celsius=33, v_init=-120, h_dt=0.01,
                 min_inter=0.1, max_inter=5000, num_pts=50, cond_st_dur=1, res_pot=-120, dur=0.1,
                 vec_pts=np.linspace(-120, 0, num=13),
                 f3cl_dur0=50, f3cl_amp0=-120, f3cl_dur1=5, f3cl_amp1=0, f3cl_dur2=1,
                 f3cl_dur3=5, f3cl_amp3=0, f3cl_dur4=5, f3cl_amp4=-120):
        self.recordTime = recordTime

        # one-compartment cell (soma)
        super().__init__(channel_name, soma_diam, soma_L, soma_nseg, soma_cm, soma_Ra, soma_ena)
        
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

            self.update_clamp_time_step()

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
