import json
from scipy.signal import find_peaks
from vm_plotter import plot_stim_volts_pair
from neuron import h





def update_mech_from_dict(mdl,dict_fn,mechs):
    with open(dict_fn) as f:
        data = f.read()
    param_dict = json.loads(data)
    for curr_sec in mdl.sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                for p_name in param_dict.keys():
                    hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} = {param_dict[p_name]}'
                    print(hoc_cmd)
                    h(hoc_cmd)
                #in case we need to go per sec:
                  #  for seg in curr_sec:
                  #      hoc_cmd = f'{curr_name}.gbar_{channel}({seg.x}) *= {wt_mul}'
                  #      print(hoc_cmd)

def update_mod_param(mdl,mechs,mltplr,gbar_name = 'gbar'):
    for curr_sec in mdl.sl:
        curr_name = h.secname(sec=curr_sec)
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                for seg in curr_sec:
                    hoc_cmd = f'{curr_name}.{gbar_name}_{curr_mech}({seg.x}) *= {mltplr}'
                    print(hoc_cmd)
                    par_value = h(f'{curr_name}.{gbar_name}_{curr_mech}({seg.x})')
                    h(hoc_cmd)
                    assigned_value = h(f'{curr_name}.{gbar_name}_{curr_mech}({seg.x})')
                    print(f'par_value before{par_value} and after {assigned_value}')
def multiply_param(mdl,mechs,p_name,multiplier):
    for curr_sec in mdl.sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} *= {multiplier}'
                print(hoc_cmd)
                h(hoc_cmd)
def offset_param(mdl,mechs,p_name,offset):
    for curr_sec in mdl.sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} += {offset}'
                print(hoc_cmd)
                h(hoc_cmd)
#### Emily's code
def update_channel(mdl, channel_name, channel, dict_fn, wt_mul, mut_mul):
    """
    channel_name: str e.g 'na16mut'
    channel: str e.g. 'na16'
    """
    with open(dict_fn) as f:
        data = f.read()
    param_dict = json.loads(data)
    for curr_sec in mdl.sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_{channel_name}({seg.x}) *= {mut_mul}'
                print(hoc_cmd)
                h(hoc_cmd)
            for p_name in param_dict.keys():
                hoc_cmd = f'{curr_name}.{p_name} = {param_dict[p_name]}'
                print(hoc_cmd)
                h(hoc_cmd)
        if h.ismembrane(channel, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_{channel}({seg.x}) *= {wt_mul}'
                print(hoc_cmd)
                h(hoc_cmd)


def update_K(mdl, channel_name, gbar_name, mut_mul):
    k_name = f'{gbar_name}_{channel_name}'
    prev = []
    for curr_sec in mdl.sl:
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


def reverse_update_K(mdl, channel_name, gbar_name, prev):
    k_name = f'{gbar_name}_{channel_name}'
    index = 0
    for curr_sec in mdl.sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = prev[index]
                h(hoc_cmd)
                index += 1

def plot_stim(mdl, amp,fn,clr='blue'):
    mdl.init_stim(amp=amp)
    Vm, I, t, stim = mdl.run_model()
    plot_stim_volts_pair(Vm, f'Step Stim {amp}pA', file_path_to_save=f'./Plots/V1/{fn}_{amp}pA',times=t,color_str=clr)
    return I

def plot_all_FIs(mdl, fis, extra_cond = False):
    for i in range(len(fis)):
        data = fis[i]
        # save multiple figures in one pdf file
        filename= f'Plots/Kexplore/FI_plots{i}.pdf'
        fig = plt.figure()
        x_axis, npeaks, name = data[0]
        plt.plot(x_axis, npeaks, label=name, color='black')
        # plot mut
        x_axis, npeaks, name = data[1]
        plt.plot(x_axis, npeaks, label=name, color='red')
        if extra_cond:
            # plot wtTTX
            x_axis, npeaks, name = data[2]
            plt.plot(x_axis, npeaks, label=name, color='black', linestyle='dashed')
            # plot mutTTX
            x_axis, npeaks, name = data[3]
            plt.plot(x_axis, npeaks, label=name, color='red', linestyle='dashed')

        plt.legend()
        plt.xlabel('Stim [nA]')
        plt.ylabel('nAPs for 500ms epoch')
        plt.title(f'FI Curve: for range {i}')
        fig.savefig(filename)

