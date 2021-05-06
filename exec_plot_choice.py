from generalized_genSim_shorten_time import *
from eval_helper import *
from fpdf import FPDF
import os

def exec_plot_choice(plot_choice, new_params = False, save_plots_as_pdf = False, 
                     WT_channel_name = 'na12mut', variant_channel_name = None):
    """Renders the plot of choice
    
    args:
        plot_choice: a string that indicates which function to plot. 
    returns:
        None
    """
    
    

    #updates parameters if given any
    if new_params:
        change_params_dict(new_params)

    #plots each function
    if plot_choice == "Activation":
        #makes subplots
        fig, axs = plt.subplots(4, 2, figsize=(10,20))
        fig.subplots_adjust(hspace=.5, wspace=.5)
    
        genAct = Activation(channel_name = WT_channel_name)
        genAct.genActivation()
        genAct.plotAllActivation_with_ax([axs[0][0], axs[1][0]], is_variant = False)
        
        if variant_channel_name:
            genAct = Activation(channel_name = variant_channel_name)
            genAct.genActivation()
            genAct.plotAllActivation_with_ax([axs[0][1], axs[1][1]], is_variant = True, channel_name = variant_channel_name)
        
    elif plot_choice == "Inactivation":
        #makes subplots
        fig, axs = plt.subplots(4, 2, figsize=(10,20))
        fig.subplots_adjust(hspace=.5, wspace=.5)
        
        genInact = Inactivation(channel_name = WT_channel_name)
        genInact.genInactivation()
        genInact.plotAllInactivation_with_ax([axs[0][0], axs[1][0]], is_variant = False)
        
        if variant_channel_name:
            genInact = Inactivation(channel_name = variant_channel_name)
            genInact.genInactivation()
            genInact.plotAllInactivation_with_ax([axs[0][1], axs[1][1]], is_variant = True, channel_name = variant_channel_name)
        
        
    elif plot_choice == "RFI":  
        genRFI = RFI()
        genRFI.genRecInactTau()
        genRFI.plotAllRFI()
        
        
    elif plot_choice == "Ramp":     
        genRamp = Ramp()
        genRamp.genRamp()
        genRamp.plotAllRamp()
        
            
    elif plot_choice == "RecInact":      
        genRFIdv = RFI_dv()
        genRFIdv.genRecInact_dv()
        genRFIdv.genRecInactTau_dv()
        genRFIdv.genRecInactTauCurve_dv()
        genRFIdv.plotRecInact_dv()
        genRFIdv.plotRecInactProcedure_dv()

    #saves the plots as pdf in the same directory
    if save_plots_as_pdf:
        plot_path = os.path.abspath(os.getcwd()) + '/Plots_Folder'
        all_plots_name = os.listdir(plot_path)

        pdf = FPDF()
        pdf.add_page()
        for image in [plot_path + '/' + f for f in all_plots_name if f.startswith(plot_choice)]:
            pdf.image(image)

        pdf.output(plot_choice + "_plot.pdf", "F")

        
    
    
