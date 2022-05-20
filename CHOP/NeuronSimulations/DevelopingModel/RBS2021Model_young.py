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
import json
param_names = ['sh','tha','qa','Ra','Rb','thi1','thi2','qd','qg','mmin','hmin','q10','Rg','Rd','thinf','qinf','vhalfs','a0s','zetas','gms','smax','vvh','vvs']




def read_params_data(fn):
    with open(fn) as f:
        data = f.read()
        js = json.loads(data)
    return js


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
    h.working_young()

def update_na12(channel_name,param_list,val_list):
    for curr_sec in sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            #h('gbar_na16mut = 1*gbar_na16mut')
            for (pname,pval) in zip(param_list,val_list):
                if pname == 'Ena':
                    continue
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{pname}_{channel_name} = {pval}'
                #print(hoc_cmd)
                h(hoc_cmd)
            
    
def init_stim(sweep_len = 800, stim_start = 100, stim_dur = 500, amp = 0.3, dt = 0.025):
    # updates the stimulation params used by the model
    # time values are in ms
    # amp values are in nA
    
    h("st.del = " + str(stim_start))
    h("st.dur = " + str(stim_dur))
    h("st.amp = " + str(amp))
    h.tstop = sweep_len
    h.dt = dt


def get_fi_curve(s_amp,e_amp,nruns,wt_data=None,ax1=None,stim_fn = None):
    all_volts = []
    npeaks = []
    x_axis = np.linspace(s_amp,e_amp,nruns)
    dt = 0.1
    stim_length = int(500/dt)
    for curr_amp in x_axis:
        if stim_fn is None:
            init_stim(amp = curr_amp)
        h.dt = 0.1
        curr_volts,_,_,_ = run_model(stim_fn = stim_fn,factor = curr_amp)
        curr_peaks,_ = find_peaks(curr_volts,height = -20)
        all_volts.append(curr_volts)
        npeaks.append(len(curr_peaks))
    print(npeaks)
    if ax1 is None:
        fig,ax1 = plt.subplots(1,1)
    ax1.plot(x_axis,npeaks,'o',color = 'black')
    ax1.set_title('FI Curve')
    ax1.set_xlabel('Stim [nA]')
    ax1.set_ylabel('nAPs for 500ms epoch')
    ax1.set_xticks(x_axis)
    if wt_data is None:
        return npeaks
    else:
        ax1.plot(x_axis,npeaks,'red')
        ax1.plot(x_axis,npeaks,'o',color = 'red')
        ax1.plot(x_axis,wt_data,'black')
    
    #plt.show()
    
    
def run_model(start_Vm = -72,stim_fn = None,factor = 1):
    timesteps = int(h.tstop/h.dt)
    if stim_fn is not None:
        stim_in = np.genfromtxt(stim_fn, dtype=np.float32)
        stim_in = stim_in * factor * 0.5
        h("st.del = 0")
        h.st.dur = 1e9
        h.st.amp = stim_in[0]
        timesteps = len(stim_in)
    h.working_young()
    h.finitialize(start_Vm)
    
    Vm = np.zeros(timesteps)
    I = {}
    I['Na'] = np.zeros(timesteps)
    I['Ca'] = np.zeros(timesteps)
    I['K'] = np.zeros(timesteps)
    stim = np.zeros(timesteps)
    t = np.zeros(timesteps)
    for i in range(timesteps):
        if stim_fn is not None:
            h.st.amp = stim_in[i]
        Vm[i] = h.cell.soma[0].v
        I['Na'][i] = h.cell.soma[0](0.5).ina
        I['Ca'][i] = h.cell.soma[0](0.5).ica
        I['K'][i] = h.cell.soma[0](0.5).ik
        stim[i] = h.st.amp 
        t[i] = i*h.dt / 1000
        h.fadvance()
        
    return Vm, I, t,stim

param_fn = '../../files_for_optimization/csv_files/mutants_parameters.txt'
var_param_dict = read_params_data(param_fn)

def run_young_model(mut,amps,fi_range,nsweeps,stim_fn = None):
    init_settings()
    fig,ficurveax = plt.subplots(1,1)
    vs_plots = []
    update_na12('na12A', param_names, var_param_dict['A_WT'])
    update_na12('na12N', param_names, var_param_dict['N_WT'])
    update_na12('na12A_Mut', param_names, var_param_dict['A_WT'])
    update_na12('na12N_Mut', param_names, var_param_dict['N_WT'])
    for cur_amp in amps:
        init_stim(amp=cur_amp)
        Vm, I, t, stim = run_model(stim_fn = stim_fn,factor = cur_amp)
        curr_ax = plot_volts(Vm, f'Step {cur_amp}nA {mut}', times=t)
        vs_plots.append(curr_ax)
    #wtnpeaks = get_fi_curve(fi_range[0],fi_range[1], nsweeps,ax1=ficurveax,stim_fn= stim_fn)
    #simulate the Mutant
    mut_a = f'A_{mut}'
    mut_n = f'N_{mut}'
    update_na12('na12A', param_names, var_param_dict['A_WT'])
    update_na12('na12N', param_names, var_param_dict['N_WT'])
    update_na12('na12A_Mut', param_names, var_param_dict[mut_a])
    update_na12('na12N_Mut', param_names, var_param_dict[mut_n])
    for cur_ax,cur_amp in zip(vs_plots,amps):
        init_stim(amp=cur_amp)
        Vm, I, t, stim = run_model(stim_fn = stim_fn,factor = cur_amp)
        plot_volts(Vm, f'Step {cur_amp}nA {mut}',axs = cur_ax, file_path_to_save = f'./Plots/{mut}_{cur_amp}_stepsynstim.pdf', times=t,color_str = 'red')
    #get_fi_curve(fi_range[0],fi_range[1],nsweeps,wt_data=wtnpeaks,ax1=ficurveax,stim_fn = stim_fn)
    #ficurveax.set_title(f'FI Curve {mut} syn_stim')
    #ficurveax.set_ylim([-1,40])
    #fig.savefig(f'./Plots/{mut}_FI_curve_syn_stim.pdf')
    return fig,ficurveax

def run_wt(factors,stim_fn = None):
    init_settings()
    fig,vax = plt.subplots(1,1)
    vs_plots = []
    update_na12('na12A', param_names, var_param_dict['A_WT'])
    update_na12('na12N', param_names, var_param_dict['N_WT'])
    update_na12('na12A_Mut', param_names, var_param_dict['A_WT'])
    update_na12('na12N_Mut', param_names, var_param_dict['N_WT'])
    for fact in factors:
        if stim_fn is None:
            init_stim(amp= fact)
        Vm, I, t, stim = run_model(stim_fn = stim_fn,factor = fact)
        plot_volts(Vm, f'Stim with {fact} factor', times=t,axs = vax)
    fig.savefig(f'./Plots/WTsyn_stim.pdf')
#run_wt([0.7])
#run_wt([1],stim_fn = '../syn_stim.csv')
#run_young_model('T400R',[0.5,1.25],[0,1.5],7)
fig,axs = run_young_model('M1770L',[0.8,1.6],[0,0],11)
# fig,ficurveax = plt.subplots(1,1)
# init_settings()
# init_stim(amp=0.75)
# Vm, I, t, stim = run_model()
# plot_stim_volts_pair(Vm, 'Step Stim 200pA', file_path_to_save='./Plots/WT_200pA',times=t)
# init_stim(amp=1.25)
# Vm, I, t, stim = run_model()
# plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/WT_500pA',times=t)
# wtnpeaks = get_fi_curve(0.75, 1.25, 11,ax1=ficurveax)


# update_na16('./params/na16_mutv1.txt')
# init_stim(amp=0.2)
# Vm, I, t, stim = run_model()
# plot_stim_volts_pair(Vm, 'Step Stim 200pA', file_path_to_save='./Plots/Mut_200pA',times=t,color_str='blue')
# init_stim(amp=0.5)
# Vm, I, t, stim = run_model()
# plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/Mut_500pA',times=t,color_str='blue')
# get_fi_curve(0.05, 0.55, 11,wt_data=wtnpeaks,ax1=ficurveax)
# fig.savefig('./Plots/FI_curves.pdf')



