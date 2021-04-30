# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:11:30 2021

@author: bensr
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
from scipy import stats
import matplotlib.colors as clr
from neuron import h
import numpy as np

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

tick_major = 6
tick_minor = 4
plt.rcParams["xtick.major.size"] = tick_major
plt.rcParams["xtick.minor.size"] = tick_minor
plt.rcParams["ytick.major.size"] = tick_major
plt.rcParams["ytick.minor.size"] = tick_minor

font_small = 12
font_medium = 13
font_large = 14
plt.rc('font', size=font_small)          # controls default text sizes
plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_small)    # legend fontsize
plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

global sl
def init_neuron():
    global sl
    h.load_file("runModel.hoc")
    soma_ref = h.root.sec
    print(soma_ref)
    soma = h.secname(sec=soma_ref)
    sl = h.SectionList()
    sl.wholetree(sec=soma_ref)
def init_stim(sweep_len = 5000, stim_start = 1000, stim_dur = 2000, amp = -0.1, dt = 0.1):
    # updates the stimulation params used by the model
    # time values are in ms
    # amp values are in nA
    
    h("st.del = " + str(stim_start))
    h("st.dur = " + str(stim_dur))
    h("st.amp = " + str(amp))
    h.tstop = sweep_len
    h.dt = dt
    
def init_settings():
    # create default model parameters to avoid loading the model
    
    h.dend_na12 =0.026145/2
    h.dend_na16 =h.dend_na12
    h.dend_k = 0.004226


    h.soma_na12 = 0.983955/10
    h.soma_na16 = h.soma_na12
    h.soma_K = 0.303472

    h.ais_na16=4
    h.ais_na12=4
    h.ais_ca = 0.000990
    h.ais_KCa = 0.007104

    h.node_na = 2

    h.axon_KP =0.973538
    h.axon_KT = 0.089259
    h.axon_K = 1.021945

    h.cell.axon[0].gCa_LVAstbar_Ca_LVAst = 0.001376286159287454
    
    #h.soma_na12 = h.soma_na12/2
    h.naked_axon_na = h.soma_na16/5
    h.navshift = -10
    h.myelin_na = h.naked_axon_na
    h.myelin_K = 0.303472
    h.myelin_scale = 10
    h.gpas_all = 3e-5
    h.cm_all = 1
    
    h.working()
    
def run_model(start_Vm = -72):

    h.finitialize(start_Vm)
    timesteps = int(h.tstop/h.dt)
    
    Vm = np.zeros(timesteps)
    I = {}
    I['Na'] = np.zeros(timesteps)
    I['Ca'] = np.zeros(timesteps)
    I['K'] = np.zeros(timesteps)
    t = np.zeros(timesteps)
    
    for i in range(timesteps):
        Vm[i] = h.cell.soma[0].v
        I['Na'][i] = h.cell.soma[0](0.5).ina
        I['Ca'][i] = h.cell.soma[0](0.5).ica
        I['K'][i] = h.cell.soma[0](0.5).ik
        t[i] = i*h.dt / 1000
        h.fadvance()
        
    return Vm, I, t   
def plot_model(line_color,fig=None):
    sweep_len = 450
    stim_dur = 300
    amp = 0.5
    dt = 0.01
    init_stim(sweep_len = sweep_len, stim_start = 100,stim_dur = stim_dur, amp = amp, dt = dt)
    Vm, I, t = run_model()
    dvdt = np.gradient(Vm)/h.dt
    if (fig==None):
        fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20,4), sharex=False, sharey=False)
    else:
        [ax1,ax2] = fig.axes
    fig_title = 'Model Run Example'
    fig.suptitle(fig_title) 

    title_txt = '{amp}nA for {stim_dur}ms'.format(amp = amp, stim_dur = stim_dur)
    ax1.set_title(title_txt) 
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Vm (mV)')

    ax2.set_title('Phase plane')
    ax2.set_xlabel('Vm (mV)')
    ax2.set_ylabel('dVdt (V/s)')

    ax1.plot(t, Vm, color = line_color)
    ax2.plot(Vm, dvdt, color = line_color)

    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    tick_major = 6
    tick_minor = 4
    plt.rcParams["xtick.major.size"] = tick_major
    plt.rcParams["xtick.minor.size"] = tick_minor
    plt.rcParams["ytick.major.size"] = tick_major
    plt.rcParams["ytick.minor.size"] = tick_minorfont_small = 12
    font_medium = 13
    font_large = 14
    plt.rc('font', size=font_small)          # controls default text sizes
    plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
    plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_small)    # legend fontsize
    plt.rc('figure', titlesize=font_large)   # fontsize of the figure title
    return fig

def update_16(yaml_fn,mod_suffix):
    with open(yaml_fn, 'r') as stream:
        param_dict = yaml.safe_load(stream)
    p_names = list(param_dict.keys())
    for curr_sec in sl:
        for p in p_names:
            p_value = param_dict[p]
            if ('Ena' in p):
                p='Ena'
                continue
            str = f'{p}_{mod_suffix}={p_value}'
            
            h(str)
    
def ko12():
    for curr_sec in sl:
        curr_sec.gbar_na12 = 0
        
def add_ttx():
    
    h.ais_na16= 0
    h.ais_na16mut= 0
    h.ais_na12= 0
    h.dend_na12 = 0
    h.dend_na16 = 0
    h.dend_na16mut = 0
    h.soma_na12 = 0
    h.soma_na16 = 0
    h.soma_na16mut = 0
    h.node_na = 0
    h.naked_axon_na = 0
    h.working()
def plot_na16_muts():
    wt_fn = '../SCN8A_WT_opt_params.yaml'
    mut_fn = '../SCN8A_MUT_opt_params.yaml'
   # update_16(wt_fn, 'na16')
   # update_16(wt_fn, 'na16mut')
    run_model()
    fig_wt = plot_model('black')
    plt.savefig('withena.pdf')
    
init_neuron()
init_settings()
fig = plot_model('black')
plot_na16_muts()

#fig = plot_model('black')

#plt.show()