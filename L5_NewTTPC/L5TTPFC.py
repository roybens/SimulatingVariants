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
def additional_settings():
        #M Current Kinetics
    #mAlpha = a_pre_exp/(1+exp((-v+a_vshift)/a_exp_div)) 
    #mBeta =  b_pre_exp*exp((-v+ b_vshift)/b_exp_div)
    #mInf = mAlpha/(mAlpha + mBeta)
    #mTau = 1/(mAlpha + mBeta)
    Im_a_pre_exp = 0.02 
    Im_a_vshift = -20
    Im_a_exp_div = 0.2
    Im_b_pre_exp = 0.01    #0.01 
    Im_b_vshift = -50      #-45
    Im_b_exp_div = 0.05556*10       #0.005556
    #BK Kinetics:
    #PROCEDURE rates( v (mV) ) {
    	#v = v + 5 (mV)
    	#minf = 1 / ( 1+exp(-(v+cvm)/ckm) )
    	#taum = (1e3) * ( ctm + 1 (s) / ( exp(-(v+cvtm1)/cktm1) + exp(-(v+cvtm2)/cktm2) ) ) / qt
    	
    	#zinf = 1 /(1 + zhalf/cai)
        #  tauz = ctauz/qt
    
    	#hinf = ch + (1-ch) / ( 1+exp(-(v+cvh)/ckh) )
    	#tauh = (1e3) * ( cth + 1 (s) / ( exp(-(v+cvth1)/ckth1) + exp(-(v+cvth2)/ckth2) ) ) / qt
    #}
    zhalf = 0.005  # mM 0.01
    cvm = 28.9 #(mV)28.9
    ckm = 6.2 #(mV)6.2
    
    ctm = 0.00150  #(s)0.000505
    cvtm1 = 86.4 #(mV)86.4
    cktm1 = -10.1 #(mV)-10.1
    cvtm2 = -33.3#(mV)-33.3
    cktm2 = 10 #(mV)10
    
    ctauz = 1 #(ms)1
    
    ch = 0.085#0.085
    cvh = 32 #(mV)32
    ckh = -5.8 #(mV)-5.8
    cth = 0.0019 #(s)0.0019
    cvth1 = 48.5 #(mV)48.5
    ckth1 = -5.2 #(mV)-5.2
    cvth2 = -54.2 #(mV)-54.2
    ckth2 = 12.9 #(mV)12.9
    #h.dend_na12 =0.0026145
    #h.dend_na16 =0.0026145
    
    h.dend_na12 = 0.0196791*0.8*0.6
    h.dend_na16 = 0.0196791*0.8*0.6
    h.soma_na12 = 0.0196791*1.5  #1.9
    h.soma_na16 = 0.0196791*1.5
    h.ais_na16=0.8*0.7*1.0   #1.1
    h.ais_na12=0.8*0.7*1.0
    
    
    h.dend_k = 0.0008452/26*0         #Kv3.1
    h.soma_K = 0.0606944/50*0         #kv3.1
    h.axon_K = 0.104389/26*0          #kv3.1
    
    h.dend_KT = 0.0178518*1
    h.soma_KT = 0.0178518*5
    h.axon_KT = 0.0178518*5  #0.0178518*40*.5
    
    KT_mtau = 1   #activation
    KT_htau = 0.5    #inactivation  0.7  before 
    
    
    #OLD KCa (SK,BK combined)
    h.soma_KCa = 0 #0.008407*0.05
    h.dend_KCa = 0 # 0.008407*0.3
    h.ais_KCa = 0 #0.0014208*0       
    
    soma_SK = 0.008407*200000000000
    dend_SK = 0.008407*200000000000
    ais_SK = 0.0014208*0
    
    soma_BK = 0.008407*30000000*1   #1.15
    dend_BK = 0.008407*30000000*1
    ais_BK = 0.0014208*0
    
    h.dend_Im =0.0000286*150*1  # all 3 started at 0.5 in FINAL 102720 KJB
    h.soma_Im =0.0000286*750*2
    h.axon_Im =0.0000286*1500*1
    
    #---super phat
    #Right now KiM is only at the dendrites 
    
    
    
    h.axon_KP =0.1947076*0.75     #-------phat  0.5
    
    
    
    
    #h.soma_KCa = 0.0016814*0      #BK
    #h.axon_KP =0.1947076
    #h.axon_KT = 0.0178518*0.1
    
    #------------------------------
    
    
    h.dend_CaT = 0.0000666*35000*1  #0.25
    h.soma_CaT = 0.0000666*350*1    #0.5
    h.soma_Ca = 0.0001988*1.2 #HVA
    
    h.ais_ca = 0.000198*0.01 #HVA
    
    h.dend_ih = 0.000100*1  #0.00005
    h.soma_ih = 0.000030*1  #0.00008
    
    h.node_na = 0.4
    
    h.gpas_all = 5e-6*20    # started at 5e-6 in FINAL
    epas_all = -88
    h.cm_all = 0.4
    
    for curr_sec in sl:
        #print(curr_sec)
        curr_sec.ena = 60
        curr_sec.e_pas = h.epas_all
        curr_sec.Ra = 100
        curr_sec.cm=h.cm_all
        curr_sec.htau_fact_K_Tst = KT_htau
        curr_sec.mtau_fact_K_Tst = KT_mtau
        curr_sec.a_pre_exp_Im = Im_a_pre_exp
        curr_sec.a_vshift_Im = Im_a_vshift
        curr_sec.a_exp_div_Im = Im_a_exp_div
        curr_sec.b_pre_exp_Im = Im_b_pre_exp
        curr_sec.b_vshift_Im = Im_b_vshift
        curr_sec.b_exp_div_Im = Im_b_exp_div
        #UPDATING BK
        
        curr_sec.cvm_bk = cvm 
        curr_sec.ckm_bk = ckm 
    
        curr_sec.ctm_bk = ctm 
        curr_sec.cvtm1_bk = cvtm1  
        curr_sec.cktm1_bk = cktm1  
        curr_sec.cvtm2_bk = cvtm2  
        curr_sec.cktm2_bk = cktm2 
        curr_sec.ctauz_bk = ctauz 
        curr_sec.ch_bk = ch  
        curr_sec.cvh_bk = cvh  
        curr_sec.ckh_bk = ckh 
        curr_sec.cth_bk = cth
        curr_sec.cvth1_bk = cvth1 
        curr_sec.ckth1_bk = ckth1 
        curr_sec.cvth2_bk = cvth2
        curr_sec.ckth2_bk = ckth2
        curr_sec.zhalf_bk = zhalf
    for curr_sec in h.cell.apical:
        #curr_sec.ehcn_Ih = -30
        curr_sec.gIhbar_Ih = h.dend_ih
        curr_sec.gSKv3_1bar_SKv3_1 = h.dend_k 
        curr_sec.gImbar_Im = h.dend_Im
        curr_sec.gbar_sk = dend_SK
        curr_sec.gbar_bk = dend_BK
    for curr_sec in h.cell.basal:
        #curr_sec.ehcn_Ih = -30
        curr_sec.gIhbar_Ih = h.dend_ih
        curr_sec.gSKv3_1bar_SKv3_1 = h.dend_k 
        curr_sec.gImbar_Im = h.dend_Im
        curr_sec.gbar_sk = dend_SK
        curr_sec.gbar_bk = dend_BK
    for curr_sec in h.cell.somatic:
        curr_sec.gImbar_Im = h.soma_Im
        curr_sec.gbar_sk = soma_SK
        curr_sec.gbar_bk = soma_BK
    for curr_sec in h.cell.axonal:
        curr_sec.gImbar_Im = h.axon_Im
        curr_sec.gbar_sk = ais_SK
        curr_sec.gbar_bk = ais_BK
    for curr_sec in sl:
        h.Rg_na12 = 0.1 #ms  0.1
        h.ar2_na12 = 0.75 #0 inactivation, 1 no inactivation   0.75
        h.Rg_na16 = 0.01 #ms  0.1
        h.ar2_na16 = 0.25 #0 inactivation, 1 no inactivation  0.25
        h.thi2_na12 = -45 #mV
        h.thi2_na16 = -45 #mV
        h.sh_na12 = 4.5  #default is 8
        h.sh_na16 = 4.5
    h.working()

    
    # Run model
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
def plot_model():
    sweep_len = 450
    stim_dur = 300
    amp = 0.15
    dt = 0.01
    init_stim(sweep_len = sweep_len, stim_start = 100,stim_dur = stim_dur, amp = amp, dt = dt)
    Vm, I, t = run_model()
    dvdt = np.gradient(Vm)/h.dt
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20,4), sharex=False, sharey=False)
    fig_title = 'Model Run Example'
    fig.suptitle(fig_title) 

    title_txt = '{amp}nA for {stim_dur}ms'.format(amp = amp, stim_dur = stim_dur)
    ax1.set_title(title_txt) 
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Vm (mV)')

    ax2.set_title('Phase plane')
    ax2.set_xlabel('Vm (mV)')
    ax2.set_ylabel('dVdt (V/s)')

    ax1.plot(t, Vm, color = 'k')
    ax2.plot(Vm, dvdt, color = 'k')

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


#TODO Michael
def update_16(params_dict):
     for curr_sec in sl:
         #iterate over all the paramters and assign the param name with the suffix of _na16 to the value 
         str = f'{param_key}_na16={param_value}'
         h(str)
    
def ko12():
    for curr_sec in sl:
        curr_sec.gbar_na12 = 0
        
def add_ttx():
    
    h.ais_na16= 0
    h.ais_na12= 0
    h.dend_na12 = 0
    h.dend_na16 = 0
    h.soma_na12 = 0
    h.soma_na16 = 0
    h.node_na = 0
    h.naked_axon_na = 0
    h.working()
init_neuron()
additional_settings()
#run_model()
plot_model()
plt.show()