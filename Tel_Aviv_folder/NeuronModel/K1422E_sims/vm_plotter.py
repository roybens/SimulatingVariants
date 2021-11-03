
import matplotlib.pyplot as plt
import numpy as np
from scalebary import add_scalebar
my_dpi = 96
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
ntimestep = 10000
dt = 0.02
def_times = np.array([dt for i in range(ntimestep)])
def_times = np.cumsum(def_times)
def cm_to_in(cm):
    return cm/2.54

def plot_stim_volts_pair(stim, volts, title_stim, title_volts, file_path_to_save=None,times=def_times):
    fig,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(7.8)),gridspec_kw={'height_ratios': [1, 8],'wspace': 0.05})
    axs[0].set_title(title_stim)
    axs[0].plot(times,stim, color='black', linewidth=0.25)
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].locator_params(axis='y', nbins=5)
    
    add_scalebar(axs[0])
    #=axs[0].set_title('Voltage Response '+title_volts)
    volts_target = volts[0]
    if len(volts)>1:
        volts_best_response = volts[1]
        axs[1].plot(times,volts_best_response, label='response', color='red',linewidth=1)
    
    
    axs[1].plot(times,volts_target, label='target', color='black',linewidth=1)
    
    axs[1].locator_params(axis='x', nbins=5)
    axs[1].locator_params(axis='y', nbins=8)
    add_scalebar(axs[1])
    
    #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    #plt.tight_layout(pad=1)
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=my_dpi, bbox_inches="tight")
    return fig,axs
        
