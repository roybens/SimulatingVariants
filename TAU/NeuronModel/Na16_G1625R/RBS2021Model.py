# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:07:44 2021

@author: bensr
"""
import argparse
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
    prev = []
    with open(dict_fn) as f:
        data = f.read()
    param_dict = json.loads(data)
    for curr_sec in sl:
        if h.ismembrane('na16mut', sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_na16mut({seg.x}) *= {mut_mul}'
                #val = h(f'{curr_name}.gbar_na16mut({seg.x})')
                #prev.append(f'{curr_name}.gbar_na16mut({seg.x}) = {val}')
                #print(hoc_cmd)
                h(hoc_cmd)
            for p_name in param_dict.keys():
                hoc_cmd = f'{curr_name}.{p_name} = {param_dict[p_name]}'
                #val = h(f'{curr_name}.{p_name}')
                #prev.append(f'{curr_name}.{p_name} = {val}')
                #print(hoc_cmd)
                h(hoc_cmd)
        if h.ismembrane('na16', sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_na16({seg.x}) *= {wt_mul}'
                #val = h(f'{curr_name}.gbar_na16({seg.x})')
                #prev.append(f'{curr_name}.gbar_na16({seg.x}) = {val}')
                #print(hoc_cmd)
                h(hoc_cmd)
    return prev

#def reverse_updates(prev):
#    for hoc_cmd in prev:
#        h(hoc_cmd)

def update_K(channel_name,gbar_name,mut_mul):
    k_name = f'{gbar_name}_{channel_name}'
    prev = []
    for curr_sec in sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.{k_name}({seg.x}) *= {mut_mul}'
                print(hoc_cmd)
                h(f'a = {curr_name}.{k_name}({seg.x})')  # get old value
                prev_var = h.a
                prev.append(f'{curr_name}.{k_name}({seg.x}) = {prev_var}')  # store old value in hoc_cmd
                h(hoc_cmd)
    return prev

def reverse_update_K(channel_name,gbar_name, prev):
    k_name = f'{gbar_name}_{channel_name}'
    index = 0
    for curr_sec in sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                #hoc_cmd = f'{curr_name}.{k_name}({seg.x}) *= {mut_mul}'
                #print(hoc_cmd)
                #h(f'a = {curr_name}.{k_name}({seg.x})')  # get old value
                #prev_var = h.a
                #prev.append(f'{curr_name}.{k_name}({seg.x}) = {prev_var}')  # store hoc_cmd
                hoc_cmd = prev[index]
                h(hoc_cmd)
                index += 1



    
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
def plot_stim(amp,fn):
    init_stim(amp=amp)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, f'Step Stim {amp}pA', file_path_to_save=f'./Plots/Kexplore/{fn}_{amp}pA',times=t,color_str='blue')
def make_fi(ranges,fn):
    fig,ficurveax = plt.subplots(1,1)
    get_fi_curve(ranges[0], ranges[1], ranges[2],ax1 = ficurveax)
    fig.savefig(f'./Plots/Kexplore/{fn}_FI.pdf')
    


def cultured_neurons_wt(extra,fi_ranges,label):
    #init_settings()
    #update_K('SKv3_1','gSKv3_1bar',1.5)
    #update_K('K_Tst','gK_Tstbar',1.5)
    #update_K('K_Pst','gK_Pstbar',1.5)
    prev = update_na16('./params/na16_mutv2.txt',2+extra,0)
    plot_stim(0.5,f'{label}_overexpressedWT{extra}')
    make_fi(fi_ranges,f'{label}_overexpressedWT{extra}')
    return prev
    
def cultured_neurons_mut(extra,fi_ranges,label):
    #init_settings()
    #update_K('SKv3_1','gSKv3_1bar',1.5)
    #update_K('K_Tst','gK_Tstbar',1.5)
    #update_K('K_Pst','gK_Pstbar',1.5)
    prev = update_na16('./params/na16_mutv2.txt',2,extra)
    plot_stim(0.5,f'{label}_overexpressedMut{extra}')
    make_fi(fi_ranges,f'{label}_overexpressedMut{extra}')
    return prev
    
    
def cultured_neurons_wtTTX(extra,fi_ranges,label):
    #init_settings()
    #update_K('SKv3_1','gSKv3_1bar',1.5)
    #update_K('K_Tst','gK_Tstbar',1.5)
    #update_K('K_Pst','gK_Pstbar',1.5)
    prev = update_na16('./params/na16_mutv2.txt',extra,0)
    plot_stim(2,f'{label}_overexpressedWT_TTX{extra}')
    make_fi(fi_ranges,f'{label}_overexpressedWT_TTX{extra}')
    return prev
    

def cultured_neurons_mutTTX(extra,fi_ranges,label):
    #init_settings()
    #update_K('SKv3_1','gSKv3_1bar',1.5)
    #update_K('K_Tst','gK_Tstbar',1.5)
    #update_K('K_Pst','gK_Pstbar',1.5)
    prev = update_na16('./params/na16_mutv2.txt',0,extra)
    plot_stim(2,f'{label}_overexpressedmut_TTX{extra}')
    make_fi(fi_ranges,f'{label}_overexpressedmut_TTX{extra}')
    return prev
    
def explore_param(ch_name,gbar_name,ranges):
    factors = np.linspace(ranges[0],ranges[1],ranges[2])
    all_prevs = []
    for curr_factor in factors:
        init_settings()
        prev = update_K(ch_name,gbar_name,curr_factor)
        label = f'{ch_name}_{curr_factor}'
        prev_wt = cultured_neurons_wt(0.5,[0.1,1,5],label)
        prev_mut = cultured_neurons_mut(0.25,[0.1, 1, 5],label)
        prev_wtTTX = cultured_neurons_wtTTX(0.5,[0.4, 2, 6],label)
        prev_mutTTX = cultured_neurons_mutTTX(0.25,[0.4, 2, 6],label)
        all_prevs.append(prev)
    print("PREVVVSS", all_prevs[0])
    reverse_update_K(ch_name,gbar_name, all_prevs[0])



    
def het_sims():
    init_settings()
    update_na16('./params/na16_mutv2.txt',1,1)
    init_stim(amp=0.5)
    Vm, I, t, stim = run_model()
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save=f'./Plots/hetrozygous_500pA',times=t,color_str='blue')


#######################
# MAIN
#######################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulated data.')
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    if args.function == 1:
        #gK_Tstbar_K_Tst
        #gK_Pstbar_
        #update_K('SKv3_1','gSKv3_1bar',2)
        #update_K('K_Tst','gK_Tstbar',2)
        #update_K('K_Pst','gK_Pstbar',2)
        #cultured_neurons_mut(0.25,[0.1, 1, 5])
        #cultured_neurons_wt(0.5,[0.1, 1, 5])
        #cultured_neurons_wtTTX(0.5,[0.4, 2, 6])
        #cultured_neurons_mutTTX(0.25,[0.4, 2, 6])

        explore_param('SKv3_1','gSKv3_1bar',[1,2,3])
        explore_param('K_Tst','gK_Tstbar',[1,2,3])
        explore_param('K_Pst','gK_Pstbar',[1,2,3])

        # works when I run one at a time, but not all at once.



#het_sims()




