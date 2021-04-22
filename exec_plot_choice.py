from generalized_genSim_shorten_time import *

def exec_plot_choice(plot_choice):
    """Renders the plot of choice
    
    args:
        plot_choice: a string that indicates which function to plot. 
    returns:
        None
    """

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
