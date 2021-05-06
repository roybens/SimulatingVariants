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
        
      
    #makes subplots
    fig, axs = plt.subplots(4, 2, figsize=(10,20))
    fig.subplots_adjust(hspace=.5, wspace=.5)
    first_col = [ax[0] for ax in axs]
    second_col = [ax[1] for ax in axs]

    #plots each function
    if plot_choice == "Activation":
        genAct = Activation(channel_name = WT_channel_name)
        genAct.genActivation()
        genAct.plotAllActivation_with_ax(first_col, is_variant = False)
        
        if variant_channel_name:
            genAct = Activation(channel_name = variant_channel_name)
            genAct.genActivation()
            genAct.plotAllActivation_with_ax(second_col, is_variant = True, channel_name = variant_channel_name)
        
    elif plot_choice == "Inactivation":
        genInact = Inactivation(channel_name = WT_channel_name)
        genInact.genInactivation()
        genInact.plotAllInactivation_with_ax(first_col, is_variant = False)
        
        if variant_channel_name:
            genInact = Inactivation(channel_name = variant_channel_name)
            genInact.genInactivation()
            genInact.plotAllInactivation_with_ax(second_col, is_variant = True, channel_name = variant_channel_name)
        
        
    elif plot_choice == "RFI":  
        genRFI = RFI(channel_name = WT_channel_name)
        genRFI.genRecInactTau()
        genRFI.plotAllRFI_with_ax(first_col, is_variant = False)
        
        if variant_channel_name:
            genRFI = RFI(channel_name = variant_channel_name)
            genRFI.genRecInactTau()
            genRFI.plotAllRFI_with_ax(second_col, is_variant = True)
        
        
    elif plot_choice == "Ramp":   
        genRamp = Ramp(channel_name = WT_channel_name)
        genRamp.genRamp()
        f, (first_col[1], first_col[2]) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
        f.add_subplot(111, frameon=False) #for shared axes labels and big title
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.title("WT" + " Ramp: Time Current Density Relation", x=0.4, y=1.1)
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Current', labelpad= 25)
        genRamp.plotAllRamp_with_ax(first_col, False, f)
        
        if variant_channel_name:
            genRamp = Ramp(channel_name = variant_channel_name)
            genRamp.genRamp()
            f, (second_col[1], second_col[2]) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
            f.add_subplot(111, frameon=False) #for shared axes labels and big title
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            plt.title(variant_channel_name + " Ramp: Time Current Density Relation", x=0.4, y=1.1)
            plt.xlabel('Time $(ms)$')
            plt.ylabel('Current', labelpad= 25)
            genRamp.plotAllRamp_with_ax(second_col, True, f, channel_name = variant_channel_name)
            

    elif plot_choice == "RecInact":      
        genRFIdv = RFI_dv(channel_name = WT_channel_name)
        genRFIdv.genRecInact_dv()
        genRFIdv.genRecInactTau_dv()
        genRFIdv.genRecInactTauCurve_dv()
        genRFIdv.plotAllRecInact_with_ax(first_col, is_variant = False)
        
        if variant_channel_name:
            genRFIdv = RFI_dv(channel_name = variant_channel_name)
            genRFIdv.genRecInact_dv()
            genRFIdv.genRecInactTau_dv()
            genRFIdv.genRecInactTauCurve_dv()
            genRFIdv.plotAllRecInact_with_ax(second_col, is_variant = True, channel_name = variant_channel_name)

        
    #saves the plot
    plt.savefig(os.path.join(os.path.split(__file__)[0], 'Plots_Folder/' + plot_choice + '_parallel_plot'))
    if save_plots_as_pdf:
        plot_path = os.path.abspath(os.getcwd()) + '/Plots_Folder'
        pdf = FPDF()
        pdf.add_page()
        pdf.image(plot_path + '/' + plot_choice + '_parallel_plot.png')
        pdf.output(plot_choice + "_plot.pdf", "F")
        
    """  
    #saves the plots as pdf in the same directory
    if save_plots_as_pdf:
        plot_path = os.path.abspath(os.getcwd()) + '/Plots_Folder'
        all_plots_name = os.listdir(plot_path)

        pdf = FPDF()
        pdf.add_page()
        for image in [plot_path + '/' + f for f in all_plots_name if f.startswith(plot_choice)]:
            pdf.image(image)

        pdf.output(plot_choice + "_plot.pdf", "F")
        """

        
    
    
