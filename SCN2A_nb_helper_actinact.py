# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:42:06 2020
 
@author: bensr
"""
import numpy as np
from neuron import h as h1
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
def sigmoid(x,x0,k):
    y = 1.0/(1.0+np.exp(-x0*(x-k)))
    return y
def init_neuron():
    h1.load_file("runModel.hoc")
    h1.dend_na12 =0.026145/2
    h1.dend_k = 0.004226
    h1.soma_na12 = 0.983955/10 
    h1.soma_K = 0.303472
    h1.node_na = 2
    h1.axon_KP =0.973538
    h1.axon_KT = 0.089259
    h1.axon_K = 1.021945
    h1.ais_na16=4
    h1.ais_na12=4
    h1.ais_ca = 0.000990
    h1.ais_KCa = 0.007104
        
     
    h1.soma_na16 = h1.soma_na12
    #h1.soma_na12 = h1.soma_na12/2
    h1.naked_axon_na = h1.soma_na16/5
    h1.navshift = -10
    h1.dend_na16 =h1.dend_na12
    h1.myelin_na = h1.naked_axon_na
    h1.myelin_K = 0.303472
    h1.myelin_scale = 10
    h1.gpas_all = 3e-5
    h1.cm_all = 1
    h1.working()
 
     
def run_sim(amp,dt):
    h1("st.del = 100")
    h1("st.dur = 500")
    h1("st.amp = " + str(amp))
    h1("tstop = 800")
    h1.finitialize(-80)
    h1.dt = dt
    timesteps = int(h1.tstop/h1.dt)
    volts = np.zeros(timesteps)
    h1.finitialize(-80)
    for i in range(timesteps):
        curr_volt = h1.cell.soma[0].v        
        volts[i] = curr_volt
        h1.fadvance()
    return volts
def get_fi_curve(s_amp,e_amp,nruns,wt_data=None):
    all_volts = []
    npeaks = []
    x_axis = np.linspace(s_amp,e_amp,nruns)
    for curr_amp in x_axis:
        curr_volts = run_sim(curr_amp,0.05)
        curr_peaks,_ = find_peaks(curr_volts,threshold = 0)
        all_volts.append(curr_volts)
        npeaks.append(len(curr_peaks))
    print(npeaks)
    fig,ax1 = plt.subplots(1,1)
    ax1.plot(x_axis,npeaks,'black')
    ax1.set_title('FI Curve')
    ax1.set_xlabel('Stim [nA]')
    ax1.set_ylabel('nAPs for 500ms epoch')
    if wt_data is None:
        return npeaks
    else:
        ax1.plot(x_axis,npeaks,'red')
        ax1.plot(x_axis,wt_data,'black')
     
def get_volts_and_dvdt(amp,wt_data=None):
    volts = run_sim(amp,0.01)
    timesteps = int(h1.tstop/h1.dt)
    times = np.linspace(0,h1.tstop,timesteps)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
    ax1.plot(times,volts,'black')
    ax1.set_title('Model response to ' + str(amp) +'nA stim')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Vm[mV]')
    dvdt = np.gradient(volts)/h1.dt
    ax2.plot(volts,dvdt,'black')
    ax2.set_title('Phase Plane')
    ax2.set_xlabel('Vm[mV]')
    ax2.set_ylabel('dV/dt')
    ax2.set_ylim([-150,700])
    if wt_data is None:
        return volts,dvdt
    else:
        wt_volts = wt_data[0]
        ax1.plot(times,volts,'red')
        ax1.plot(times,wt_volts,'black')
        wt_dvdt = wt_data[1]
        ax2.plot(volts,dvdt,'red')
        ax2.plot(wt_volts,wt_dvdt,'black') 
    plt.show()
 
def gen_act_inact(v05act,slopeact,v05inact,slopeinact):
    xact = np.array(range(-120,40,10))
    yact = sigmoid(xact,slopeact,v05act)
    xinact = np.array(range(-120,0,10))
    yinact = sigmoid(xinact,slopeinact,v05inact)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
    # ax1.plot(xact,yact)
    # ax1.set_title('Activation')
    # ax1.set_xlabel('Vm')
    # ax2.plot(xinact,yinact)
    # ax2.set_title('Inactivation')
    # ax2.set_xlabel('Vm')
    # plt.show()
    return xinact, yinact, xact, yact
 
def make_a_het():
    h1.ais_na12=h1.ais_na12/2
    h1.soma_na12 = h1.soma_na12/2
    h1.dend_na12 = h1.dend_na12/2
    h1.working()
# init_neuron()
#
#a,b = get_volts_and_dvdt(0.5)
#make_a_het()
#get_volts_and_dvdt(0.5,[a,b])
def update_na12_mut_params(params):
    params = list(params.keys())
    for s in h1.allsec():    
        if h1.ismembrane("na12_mut"):
            for p in params:
                print (p + '=' + str(params[p]))
                h1(p + '=' + str(params[p]))
    h1.working()
        



                 