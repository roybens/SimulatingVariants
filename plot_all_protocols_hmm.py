import generalized_genSim_shorten_time_HMM as ggsdHMM
import eval_helper as eh
import HMM_plotter as plotter

p = [1.6145008130686316,
 1.2702355752969856,
 0.2856140201135051,
 2.000672353749617,
 159.19293105141264,
 0.8882089670901088,
 1.54307338742142,
 4.835533385345919,
 184.46766214071704,
 0.6193119174876813,
 8.851518497666747,
 0.07019281223744751,
 46.30970872218895,
 12.027049656918223,
 1.0303204433640094,
 0.05027526734333132,
 1791.9670172949814,
 1.3053734595552096,
 20.37380422148677,
 -9.174778056184731,
    800]

p_1 = [251.13271452152281, 0.10437122947276659, 0.12142361782233926, 0.7963678123046358, 87.7432499817526, 0.2181806981554227, 0.9354158757810526, 0.13560383511717164, 495.71666612993937, 0.015815418422276677, 6.354865560447292, 0.0038834888710251834, 12.863770137241847, 2.806668955412046, 0.6863666713200415, 3.9607925155942967, 1519.7510537871524, 0.15726941262882127, 9.740061519725462, 2.449259386085319,800]


#act_object = ggsdHMM.Activation(channel_name='na12mut8st', step=5)
#eh.change_params(p, scaled=False, is_HMM=True,sim_obj =  act_object)
#plotter.make_act_plots(p, "A427D", "./csv_files/mutant_protocols.csv", None, "./Plots_Folder/Testing_Act.pdf", is_HMM = True, channel_name = "na12mut8st")
#plotter.make_inact_plots(p, "A427D", "./csv_files/mutant_protocols.csv", None, "./Plots_Folder/Testing_Inact.pdf", is_HMM = True, channel_name = "na12mut8st")
plotter.make_recov_plots(p, "A427D", "./csv_files/mutant_protocols.csv", None, "./Plots_Folder/Testing_Inact.pdf", is_HMM = True, channel_name = "na12mut8st")

