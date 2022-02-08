# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:07:44 2021

@author: bensr
"""

import numpy as np
from vm_plotter import *
from neuron import h
import json
from scipy.signal import find_peaks
h.load_file("runModel.hoc")
soma_ref = h.root.sec
soma = h.secname(sec=soma_ref)
sl = h.SectionList()
sl.wholetree(sec=soma_ref)
def init_settings(nav12=1,
                  nav16=1,
                  dend_nav12=1, 
                  soma_nav12=1, 
                  ais_nav12=1, 
                  dend_nav16=1, 
                  soma_nav16=1,
                  ais_nav16=1, 
                  axon_Kp=1,
                  axon_Kt =1,
                  axon_K=1,
                  soma_K=1,
                  dend_K=1,
                  gpas_all=1):

    h.dend_na12 = 0.026145/2 
    h.dend_na16 = h.dend_na12 
    h.dend_k = 0.004226 * soma_K


    h.soma_na12 = 0.983955/10 
    h.soma_na16 = h.soma_na12 
    h.soma_K = 0.303472 * soma_K

    h.ais_na16 = 4 
    h.ais_na12 = 4 
    h.ais_ca = 0.000990
    h.ais_KCa = 0.007104

    h.node_na = 2

    h.axon_KP = 0.973538 * axon_Kp
    h.axon_KT = 0.089259 * axon_Kt
    h.axon_K = 1.021945 * axon_K

    h.cell.axon[0].gCa_LVAstbar_Ca_LVAst = 0.001376286159287454
    
    #h.soma_na12 = h.soma_na12/2
    h.naked_axon_na = h.soma_na16/5
    h.navshift = -10
    h.myelin_na = h.naked_axon_na
    h.myelin_K = 0.303472
    h.myelin_scale = 10
    h.gpas_all = 3e-5 * gpas_all
    h.cm_all = 1
    
    
    h.dend_na12 = h.dend_na12 * nav12 * dend_nav12
    h.soma_na12 = h.soma_na12 * nav12 * soma_nav12
    h.ais_na12 = h.ais_na12 * nav12 * ais_nav12
    
    h.dend_na16 = h.dend_na16 * nav16 * dend_nav16
    h.soma_na16 = h.soma_na16 * nav16 * soma_nav16
    h.ais_na16 = h.ais_na16 * nav16 * ais_nav16
    h.working()

def update_na16(dict_fn,wt_mul,mut_mul):
    with open(dict_fn) as f:
        data = f.read()
    param_dict = json.loads(data)
    for curr_sec in sl:
        if h.ismembrane('na16mut', sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_na16mut({seg.x}) *= {mut_mul}'
                #print(hoc_cmd)
                h(hoc_cmd)
            for p_name in param_dict.keys():
                hoc_cmd = f'{curr_name}.{p_name} = {param_dict[p_name]}'
                #print(hoc_cmd)
                h(hoc_cmd)
        if h.ismembrane('na16', sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_na16({seg.x}) *= {wt_mul}'
                #print(hoc_cmd)
                h(hoc_cmd)
            
def update_K(channel_name,gbar_name,mut_mul):
    k_name = f'{gbar_name}_{channel_name}'
    for curr_sec in sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.{k_name}({seg.x}) *= {mut_mul}'
                print(hoc_cmd)
                h(hoc_cmd)
            
    
def init_stim(sweep_len = 800, stim_start = 100, stim_dur = 500, amp = 0.3, dt = 0.1):
    # updates the stimulation params used by the model
    # time values are in ms
    # amp values are in nA
    
    h("st.del = " + str(stim_start))
    h("st.dur = " + str(stim_dur))
    h("st.amp = " + str(amp))
    h.tstop = sweep_len
    h.dt = dt


def get_fi_curve(s_amp,e_amp,nruns,wt_data=None,ax1=None):
    all_volts = []
    npeaks = []
    x_axis = np.linspace(s_amp,e_amp,nruns)
    stim_length = int(600/dt)
    for curr_amp in x_axis:
        init_stim(amp = curr_amp)
        curr_volts,_,_,_ = run_model(dt=0.5)
        curr_peaks,_ = find_peaks(curr_volts[:stim_length],height = -20)
        all_volts.append(curr_volts)
        npeaks.append(len(curr_peaks))
    print(npeaks)
    if ax1 is None:
        fig,ax1 = plt.subplots(1,1)
    ax1.plot(x_axis,npeaks,'black')
    ax1.set_title('FI Curve')
    ax1.set_xlabel('Stim [nA]')
    ax1.set_ylabel('nAPs for 500ms epoch')
    if wt_data is None:
        return npeaks
    else:
        ax1.plot(x_axis,npeaks,'blue')
        ax1.plot(x_axis,wt_data,'black')
    plt.show()
    
def run_model(start_Vm = -72,dt= 0.1):

    h.finitialize(start_Vm)
    timesteps = int(h.tstop/h.dt)
    
    Vm = np.zeros(timesteps)
    I = {}
    I['Na'] = np.zeros(timesteps)
    I['Ca'] = np.zeros(timesteps)
    I['K'] = np.zeros(timesteps)
    stim = np.zeros(timesteps)
    t = np.zeros(timesteps)
    
    for i in range(timesteps):
        Vm[i] = h.cell.soma[0].v
        I['Na'][i] = h.cell.soma[0](0.5).ina
        I['Ca'][i] = h.cell.soma[0](0.5).ica
        I['K'][i] = h.cell.soma[0](0.5).ik
        stim[i] = h.st.amp 
        t[i] = i*h.dt / 1000
        h.fadvance()
        
    return Vm, I, t,stim

def cultured_neurons_wt(extra):
    init_settings()
    update_K('SKv3_1','gSKv3_1bar',1.5)
    update_K('K_Tst','gK_Tstbar',1.5)
    update_K('K_Pst','gK_Pstbar',1.5)
    update_na16('./params/na16_mutv2.txt',2+extra,0)
    init_stim(amp=0.5)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/overexpressedWT{extra}_500pA',times=t,color_str='blue')

def cultured_neurons_mut(extra):
    init_settings()
    update_K('SKv3_1','gSKv3_1bar',1.5)
    update_K('K_Tst','gK_Tstbar',1.5)
    update_K('K_Pst','gK_Pstbar',1.5)
    update_na16('./params/na16_mutv2.txt',2,extra)
    init_stim(amp=0.5)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/overexpressedMut{extra}_500pA',times=t,color_str='blue')
    
def cultured_neurons_wtTTX(extra):
    init_settings()
    update_K('SKv3_1','gSKv3_1bar',1.5)
    update_K('K_Tst','gK_Tstbar',1.5)
    update_K('K_Pst','gK_Pstbar',1.5)
    update_na16('./params/na16_mutv2.txt',extra,0)
    init_stim(amp=1)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/overexpressed_TTX_WT{extra}_500pA',times=t,color_str='blue')
    #fig,ficurveax = plt.subplots(1,1)
    #get_fi_curve(0.1, 1, 5)
    #fig.savefig('./Plots/FI_curvesWT_TTX.pdf')

def cultured_neurons_mutTTX(extra):
    init_settings()
    update_K('SKv3_1','gSKv3_1bar',1.5)
    update_K('K_Tst','gK_Tstbar',1.5)
    update_K('K_Pst','gK_Pstbar',1.5)
    update_na16('./params/na16_mutv2.txt',0,extra)
    init_stim(amp=1)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/overexpressed_TTX_Mut{extra}_500pA',times=t,color_str='blue')
    #fig,ficurveax = plt.subplots(1,1)
    #get_fi_curve(0.1, 1, 5)
    #fig.savefig('./Plots/FI_curvesMut_TTX.pdf')

def het_sims():
    init_settings()
    update_na16('./params/na16_mutv2.txt',1,1)
    init_stim(amp=0.5)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/hetrozygous_500pA',times=t,color_str='blue')

#gK_Tstbar_K_Tst
#gK_Pstbar_
#update_K('SKv3_1','gSKv3_1bar',2)
#update_K('K_Tst','gK_Tstbar',2)
#update_K('K_Pst','gK_Pstbar',2)
cultured_neurons_mut(0.25)
cultured_neurons_wt(0.5)
cultured_neurons_wtTTX(0.5)
cultured_neurons_mutTTX(0.25)




#het_sims()



