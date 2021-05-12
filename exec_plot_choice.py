from generalized_genSim_shorten_time import *
from eval_helper import *
from fpdf import FPDF
import os
import yaml

def exec_plot_choice(plot_choice, WT_params_path, variant_params_path = None, 
                     save_plots_as_pdf = False, MT_name = "WT", mutant_name = None):
    """Renders the plot of choice
    
    args:
        plot_choice: a string that indicates which function to plot. 
    returns:
        None
    """
    
    
    with open(WT_params_path) as f:
        # use safe_load instead load
        WT_params = yaml.safe_load(f)
        
    if variant_params_path:
        with open(variant_params_path) as f:
            # use safe_load instead load
            variant_params = yaml.safe_load(f)
    else:
        variant_params = None
    
    #makes subplots
    fig, axs = plt.subplots(4, 1, figsize=(5,20))
    fig.subplots_adjust(hspace=.5, wspace=.5)
    
    
    for cur_params in ["WT_params", "variant_params"]: 
        #updates parameters
        if cur_params == "WT_params":
            change_params_dict_gen(WT_params)
        if cur_params == "variant_params":
            if variant_params:
                change_params_dict_gen(variant_params)
            else: 
                break
        

        #plots each function
        if plot_choice == "Activation":
            genAct = Activation()
            genAct.genActivation()
            genAct.plotAllActivation_with_ax(axs, cur_params)

        elif plot_choice == "Inactivation":
            genInact = Inactivation()
            genInact.genInactivation()
            genInact.plotAllInactivation_with_ax(axs, cur_params)

        elif plot_choice == "RFI":  
            genRFI = RFI()
            genRFI.genRecInactTau()
            genRFI.plotAllRFI_with_ax(axs, cur_params)


        elif plot_choice == "Ramp":   
            genRamp = Ramp()
            genRamp.genRamp()
            genRamp.plotAllRamp_with_ax(axs, cur_params)



        elif plot_choice == "RecInact":      
            genRFIdv = RFI_dv()
            genRFIdv.genRecInact_dv()
            genRFIdv.genRecInactTau_dv()
            genRFIdv.genRecInactTauCurve_dv()
            genRFIdv.plotAllRecInact_with_ax(axs, cur_params)
            
        elif plot_choice == "test": 
            genRFIdv = RFI_dv()
            genRFIdv.genRecInact_dv()
            genRFIdv.genRecInactTau_dv()
            genRFIdv.genRecInactTauCurve_dv()
            genRFIdv.plotRecInact_dv()
            genRFIdv.plotRecInactProcedure_dv()


        
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

        
    
    
