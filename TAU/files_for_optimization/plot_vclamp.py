# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:17:03 2021

@author: bensr
"""

import eval_helper_na12 as eh16
import json
mutant_protocol_csv_name = './csv_files/mutant_protocols_CHOPv2.csv'
#param_names = ['sh','tha','qa','Ra','Rb','thi1','thi2','qd','qg','mmin','hmin','q10','Rg','Rd','thinf','qinf','vhalfs','a0s','zetas','gms','smax','vvh','vvs','Ena']
def read_params_data():
    with open('./csv_files/mutants_parameters.txt') as f:
        data = f.read()
        js = json.loads(data)
    return js
def plot_vclamp_for_mut(mut_data,mut_name,wt_name):
    
    """
    mut_data: a set of variant parameters
    mut_name: the name of the mutant to plot, key of mut_data
    wt_name: the name of the WT to compare to, key of mut_data    
    """
    eh16.make_act_plots(mut_data[mut_name],mut_name,mutant_protocol_csv_name,param_values_wt = mut_data[wt_name],filename = f'./Plots_Folder/Activation_{mut_name}.pdf')
    eh16.make_inact_plots(mut_data[mut_name],mut_name,mutant_protocol_csv_name,param_values_wt = mut_data[wt_name],filename = f'./Plots_Folder/Inactivation_{mut_name}.pdf')
eh16.set_channel("na12")
mut_data = read_params_data()


plot_vclamp_for_mut(mut_data, 'A_T400R', 'A_WT')
plot_vclamp_for_mut(mut_data, 'N_T400R', 'N_WT')
plot_vclamp_for_mut(mut_data, 'A_I1640N', 'A_WT')
plot_vclamp_for_mut(mut_data, 'A_M1770L', 'A_WT')
plot_vclamp_for_mut(mut_data, 'N_M1770L', 'N_WT')