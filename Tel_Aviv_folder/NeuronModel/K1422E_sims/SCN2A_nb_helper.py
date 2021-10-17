# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:42:06 2020
 
@author: bensr
"""
import numpy as np
from neuron import h as h1
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
param_wt =  [0.02, 5.410268511568426, 7.688903669707805, 0.32811950317048455, 0.10008755035290941, 0.026568829845209807, -28.758051932013153, -37.65073337128046, -48.47847648558847]
param_F1574L = [0.02, 4.902035892042227, 7.226546233728067, 0.37867994702385005, 0.1000986379143968, 0.019403133227518025, -24.494927825744984, -31.877363364206264, -50.00697393710147]
param_M1879T = [0.02, 4.326933589039116, 5.660664333009877, 0.3194042012313818, 0.06674512918346695, 0.022044408026494475, -20.61768795320895, -20.06187562599646, -36.543988017567024]

def sigmoid(x,x0,k):
    y = 1.0/(1.0+np.exp(-x0*(x-k)))
    return y
def init_neuron():
    h1.load_file("runModel.hoc")
    h1.dend_na12 =0.026145/2
    h1.dend_k = 0.004226
    h1.soma_na12 = 0.983955/7 
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


def init_neuron_young():
    h1.load_file("runModel.hoc")
    h1.dend_na12 =0.026145/2
    h1.dend_k = 0.004226
    h1.soma_na12 = 0.983955/10 
    h1.soma_K = 0.303472
    h1.node_na = 2
    h1.axon_KP =0.973538
    h1.axon_KT = 0.089259
    h1.axon_K = 1.021945
    h1.ais_na16=5
    h1.ais_na12=5
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
    h1.working_young()
     
def run_sim(amp,dt):
    h1("st.del = 100")
    h1("st.dur = 300")
    h1("st.amp = " + str(amp))
    h1("tstop = 800")
    h1.finitialize(-73)
    h1.dt = dt
    timesteps = int(h1.tstop/h1.dt)
    volts = np.zeros(timesteps)
    h1.finitialize(-73)
    for i in range(timesteps):
        curr_volt = h1.cell.soma[0].v        
        volts[i] = curr_volt
        h1.fadvance()
    return volts
def get_fi_curve(s_amp,e_amp,nruns,wt_data=None,ax1=None):
    all_volts = []
    npeaks = []
    x_axis = np.linspace(s_amp,e_amp,nruns)
    for curr_amp in x_axis:
        curr_volts = run_sim(curr_amp,0.1)
        curr_peaks,_ = find_peaks(curr_volts,height = -20)
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
        ax1.plot(x_axis,npeaks,'red')
        ax1.plot(x_axis,wt_data,'black')
    plt.show()


     
def get_volts_and_dvdt(amp,wt_data=None,ax1=None,ax2=None):
    volts = run_sim(amp,0.01)
    timesteps = int(h1.tstop/h1.dt)
    times = np.linspace(0,h1.tstop,timesteps)
    if ax1 is None:
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
    ax2.set_ylim([-100,600])
    ax2.set_xlim([-58,50])
    #ax2.set_ylim([-150,700])
    if wt_data is None:
        return volts,dvdt
    else:
        wt_volts = wt_data[0]
        ax1.plot(times,volts,'red')
        ax1.plot(times,wt_volts,'black')
        wt_dvdt = wt_data[1]
        ax2.plot(volts,dvdt,'red')
        ax2.plot(wt_volts,wt_dvdt,'black') 
    if ax1 is None:
        return fig
    


def get_volts(amp,wt_data=None,ax1=None):
    volts = run_sim(amp,0.01)
    timesteps = int(h1.tstop/h1.dt)
    times = np.linspace(0,h1.tstop,timesteps)
    if ax1 is None:
        fig,ax1 = plt.subplots(1,1,figsize=(6,3))
    ax1.plot(times,volts,'black')
    ax1.set_title('Model response to ' + str(amp) +'nA stim')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Vm[mV]')
    if wt_data is not  None:
        wt_volts = wt_data
        ax1.plot(times,volts,'red')
        ax1.plot(times,wt_volts,'black')
    plt.show()
    return volts

def get_dvdt(volts,wt_data=None,ax2=None):
    dvdt = np.gradient(volts)/h1.dt
    if ax2 is None:
        fig,ax2 =  plt.subplots(1,1,figsize=(6,3))
    ax2.plot(volts,dvdt,'black')
    ax2.set_title('Phase Plane')
    ax2.set_xlabel('Vm[mV]')
    ax2.set_ylabel('dV/dt')
    ax2.set_ylim([-100,400])
    ax2.set_xlim([-60,60])
    if wt_data is not  None:
        wt_volts = wt_data[0]
        wt_dvdt = wt_data[1]
        ax2.plot(wt_volts,wt_dvdt,'black')
        ax2.plot(volts,dvdt,'red')
    return dvdt
     
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
def get_mech_list():
    mech_names = []
    mechlist = h1.MechanismType(0)  # object that contains all mechanism names not sure we need this
    for i in range(int(mechlist.count())):
        s = h1.ref('')  # string reference to store mechanism name
        mechlist.select(i)
        mechlist.selected(s)
        mech_names.append(s[0])
    return mech_names
def update_na12_mut_params(params):
    params_l = list(params.keys())
    for s in h1.allsec():
        if h1.ismembrane("na12mut"):
            for p in params_l:
                #print (p + '=' + str(params[p]))
                h1(p + '=' + str(params[p]))
                
    h1.working()
    h1.finitialize()
def test_params():
    init_neuron()
    start_amp = 0
    end_amp = 1
    nruns = 3
    vclamp_params_orig = {'tha_na12mut':-30,'qa_na12mut':7.2,'thinf_na12mut':-45,'qinf_na12mut':7}
    WT_fi_curve = get_fi_curve(start_amp,end_amp,nruns)   
    for i in range(5):
        i+=1
        fig,ax = plt.subplots(4,2,figsize=(24,12))
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['tha_na12mut'] += i*10
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[0][0])
        ax[0][0].title.set_text('tha_na12mut +' + str(i*10)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['tha_na12mut'] -= i*10
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[0][1])
        ax[0][1].title.set_text('tha_na12mut -' + str(i*10)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['thinf_na12mut'] += i*10
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[1][0])
        ax[1][0].title.set_text('thinf_na12mut +' + str(i*10)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['thinf_na12mut'] -= i*10
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[1][1])
        ax[1][1].title.set_text('thinf_na12mut -' + str(i*10)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['qa_na12mut'] += i*1
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[2][0])
        ax[2][0].title.set_text('qa_na12mut +' + str(i*1)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['qa_na12mut'] -= i*1
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[2][1])
        ax[2][1].title.set_text('qa_na12mut -' + str(i*1)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['qinf_na12mut'] += i*1
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[3][0])
        ax[3][0].title.set_text('qinf_na12mut +' + str(i*1)) 
        
        curr_vclamp_params = vclamp_params_orig.copy()
        curr_vclamp_params['qinf_na12mut'] -= i*1
        update_na12_mut_params(curr_vclamp_params)
        get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[3][1])
        ax[3][1].title.set_text('qinf_na12mut -' + str(i*1)) 
        
        fig.savefig('exp_' + str(i) + '.pdf')
    plt.show()
def update_na12_mut_params_from_list(new_params):
    
    """
    Change the channel params to the new params.
    """
    currh =  h1
    currh.mmin_na12mut = new_params[0] 
    currh.qa_na12mut = new_params[1] 
    currh.qinf_na12mut = new_params[2] 
    currh.Ra_na12mut = new_params[3] 
    currh.Rb_na12mut = new_params[4] 
    currh.Rd_na12mut = new_params[5] 
    currh.tha_na12mut = new_params[6] 
    currh.thi1_na12mut = new_params[7] 
    currh.thinf_na12mut = new_params[8]
    currh.thi2_na12mut = -30
    h1.working()
    return

def update_na12_mut_params_from_list_young(new_params):
    
    """
    Change the channel params to the new params.
    """
    currh =  h1
    currh.mmin_na12mut = new_params[0] 
    currh.qa_na12mut = new_params[1] 
    currh.qinf_na12mut = new_params[2] 
    currh.Ra_na12mut = new_params[3] 
    currh.Rb_na12mut = new_params[4] 
    currh.Rd_na12mut = new_params[5] 
    currh.tha_na12mut = new_params[6] 
    currh.thi1_na12mut = new_params[7] 
    currh.thinf_na12mut = new_params[8]
    currh.thi2_na12mut = -30
    
    currh.mmin_na1216mut = new_params[0] 
    currh.qa_na1216mut = new_params[1] 
    currh.qinf_na1216mut = new_params[2] 
    currh.Ra_na1216mut = new_params[3] 
    currh.Rb_na1216mut = new_params[4] 
    currh.Rd_na1216mut = new_params[5] 
    currh.tha_na1216mut = new_params[6] 
    currh.thi1_na1216mut = new_params[7] 
    currh.thinf_na1216mut = new_params[8]
    currh.thi2_na1216mut = -30
    h1.working_young()
    return

def plot_all_for_params(params):
    init_neuron()
    update_na12_mut_params_from_list(param_wt)
    start_amp = 0
    end_amp = 1
    nruns = 11
    fig,ax = plt.subplots(2,2,figsize=(24,12))
    
    WT_fi_curve = get_fi_curve(start_amp,end_amp,nruns)
    volts_05_wt = get_volts(0.5,None,ax[0][1])
    volts_1_wt = get_volts(1,None,ax[1][0])
    dvdt_1_wt = get_dvdt(volts_1_wt,None,ax[1][1])
    
    update_na12_mut_params_from_list(params)
    
    get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[0][0])
    get_volts(0.5,volts_05_wt,ax[0][1])
    volts_1_mut = get_volts(1,volts_1_wt,ax[1][0])
    
    get_dvdt(volts_1_mut,[volts_1_wt,dvdt_1_wt],ax[1][1])
    plt.show()
    fig.savefig('m1879Adult.pdf')
    fig.savefig('m1879Adult.svg')

def plot_all_for_params_young(params):
    init_neuron_young()
    update_na12_mut_params_from_list_young(param_wt)
    start_amp = .5
    end_amp = 1.5
    nruns = 11
    fig,ax = plt.subplots(2,2,figsize=(24,12))
    
    WT_fi_curve = get_fi_curve(start_amp,end_amp,nruns)
    volts_15_wt = get_volts(1.5,None,ax[0][1])
    volts_1_wt = get_volts(1,None,ax[1][0])
    #dvdt_15_wt = get_dvdt(volts_15_wt,None,ax[1][1])
    
    update_na12_mut_params_from_list_young(params)
    
    get_fi_curve(start_amp,end_amp,nruns,WT_fi_curve,ax[0][0])
    volts_15_mut = get_volts(1.5,volts_15_wt,ax[0][1])
    volts_1_mut = get_volts(1,volts_1_wt,ax[1][0])
    
    #get_dvdt(volts_15_mut,[volts_15_wt,dvdt_15_wt],ax[1][1])
    plt.show()
    fig.savefig('m1879_youngv2.pdf')
    fig.savefig('m1879_youngv2.svg',transparent=True)

def plot_exp_and_sim_dvdt():
    init_neuron()
    sim_vs = get_volts(0.25)
    exp_v = np.loadtxt('./Volts/exp_data_from_an.txt')
    exp_dvdt = np.gradient(exp_v)/0.02
    sim_dvdt =  np.gradient(sim_vs)/h1.dt
    fig,ax2 =  plt.subplots(1,1,figsize=(6,3))
    ax2.plot(exp_v,exp_dvdt,'black')
    ax2.plot(sim_vs,sim_dvdt,'red')
    ax2.set_title('Phase Plane')
    ax2.set_xlabel('Vm[mV]')
    ax2.set_ylabel('dV/dt')
    ax2.set_ylim([-100,600])
    ax2.set_xlim([-60,60])
    fig.savefig('dvdt_exp_sim.pdf', transparent=True)
  
def make_dvdt_figure(param1,param2):
    init_neuron()
    update_na12_mut_params_from_list(param1)
    volts = get_volts(1)
    dvdt =  np.gradient(volts)/h1.dt
    update_na12_mut_params_from_list(param2)
    volts2 = get_volts(1)
    dvdt2 =  np.gradient(volts2)/h1.dt
    fig,ax2 =  plt.subplots(1,1,figsize=(6,3))
    ax2.plot(volts,dvdt,'black')
    ax2.plot(volts2,dvdt2,'red')
    ax2.set_title('Phase Plane')
    ax2.set_xlabel('Vm[mV]')
    ax2.set_ylabel('dV/dt')
    ax2.set_ylim([-100,700])
    ax2.set_xlim([-60,60])
    fig.savefig('dvdt_adult_m1879.svg', transparent=True)

def make_dvdt_figure_young(param1,param2):
    init_neuron_young()
    update_na12_mut_params_from_list_young(param1)
    volts = get_volts(1.5)
    dvdt =  np.gradient(volts)/h1.dt
    update_na12_mut_params_from_list_young(param2)
    volts2 = get_volts(1.5)
    dvdt2 =  np.gradient(volts2)/h1.dt
    fig,ax2 =  plt.subplots(1,1,figsize=(6,3))
    ax2.plot(volts,dvdt,'black')
    ax2.plot(volts2,dvdt2,'red')
    ax2.set_title('Phase Plane')
    ax2.set_xlabel('Vm[mV]')
    ax2.set_ylabel('dV/dt')
    ax2.set_ylim([-100,400])
    ax2.set_xlim([-60,60])
    fig.savefig('dvdt_young_m1879.svg', transparent=True)

#plot_all_for_params_young(param_M1879T)
#plot_all_for_params(param_M1879T)
#vclamp_params = {'tha_na12mut':-50,'qa_na12mut':7.2,'thinf_na12mut':-45,'qinf_na12mut':7}
#plot_all_for_params(vclamp_params)
make_dvdt_figure_young(param_wt,param_M1879T)
        
#test_params()
#update_na12_mut_params(vclamp_params)
#h1.finitialize()
#get_fi_curve(start_amp,end_amp,nruns)
#
#                 )
#plot_exp_and_sim_dvdt()
#init_neuron()
#start_amp = 0
#end_amp = 0.5
#nruns = 11
#get_fi_curve(start_amp,end_amp,nruns)