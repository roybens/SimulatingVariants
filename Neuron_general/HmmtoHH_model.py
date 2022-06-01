from NeuronModelClass import NeuronModel
from NrnHelper import *
import matplotlib.pyplot as plt

params_folder = './params'

def make_wt():
    l5mdl = NeuronModel()
    fig, ficurveax = plt.subplots(1, 1)
    l5mdl.h.working()
    mechs = ['na16']
    update_mod_param(l5mdl, mechs, 2, gbar_name='gbar')
    mechs = ['na16mut']
    update_mod_param(l5mdl, mechs, 0, gbar_name='gbar')
    l5mdl.init_stim(amp=0.5)
    Vm, I, t, stim = l5mdl.run_model(dt=0.1)
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/WT_500pA', times=t)


def na16_het_hmm():
    l5mdl = NeuronModel()
    fig, ficurveax = plt.subplots(1, 1)
    mechs = ['na16mut']
    dict_fn = f'{params_folder}/na16WT.txt'
    update_mech_from_dict(l5mdl, dict_fn, mechs)
    l5mdl.h.working()
    update_mod_param(l5mdl, mechs, 0.05, gbar_name='gbar')
    l5mdl.init_stim(amp=0.5)
    Vm, I, t, stim = l5mdl.run_model(dt=0.1)
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/HMM_het_500pA', times=t)
def na16_hmm():
    l5mdl = NeuronModel()
    fig, ficurveax = plt.subplots(1, 1)
    mechs = ['na16mut']
    dict_fn = f'{params_folder}/na16WT.txt'
    update_mech_from_dict(l5mdl, dict_fn, mechs)
    l5mdl.h.working()
    update_mod_param(l5mdl, mechs, 0.2, gbar_name='gbar')
    mechs = ['na16']
    update_mod_param(l5mdl, mechs, 0, gbar_name='gbar')
    l5mdl.init_stim(amp=0.5)
    Vm, I, t, stim = l5mdl.run_model(dt=0.1)
    plot_stim_volts_pair(Vm, 'Step Stim 500pA', file_path_to_save='./Plots/HMM_na16_500pA', times=t)


#make_wt()
#na16_het_hmm()
na16_hmm()