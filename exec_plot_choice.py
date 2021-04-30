from generalized_genSim_shorten_time import *
from eval_helper import *
from fpdf import FPDF
import os

def exec_plot_choice(plot_choice, new_params = False, save_plots_as_pdf = False):
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
        genAct = Activation()
        genAct.genActivation()
        genAct.plotAllActivation()
    elif plot_choice == "Inactivation":
        genInact = Inactivation()
        genInact.genInactivation()
        genInact.plotAllInactivation()
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

    
