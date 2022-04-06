import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
import bluepyopt as bpop
import curve_fitting as cf
import bluepyopt.deapext.algorithms as algo
import generalized_genSim_shorten_time as ggsd
import vclamp_evaluator_HMM as vcl_ev
import pickle
import time
from deap import tools
import multiprocessing
import eval_helper_na12 as eh12
import numpy as np
import bluepyopt as bpop
import matplotlib.pyplot as plt

offspring_size = 2000
num_generations = 250
output_log_file_name = 'jinan_fitting_result_T400RAneo_interactive.txt'
param_range_file = "./csv_files/param_stats_wide_na12.csv"
mutant_name = "T400RAneo"
mutant_protocol_csv_name = './csv_files/mutant_protocols_CHOP.csv'
initial_baseline_parameters = eh12.get_wt_params_na12()


peak_amp_dict = {"T400RAdult": 0.645, "I1640NAdult": 0.24, "m1770LAdult": 0.4314, "neoWT": 0.748, "T400RAneo": 0.932, "I1640NNeo": 0.28, "m1770LNeo": 1}

class Vclamp_evaluator(bpop.evaluators.Evaluator):
    #with peak_amp

    def __init__(self, scaled):
        
        eh12.set_channel()
        
        self.scaled = scaled
        
        # (val, min, max)
        param_range_dict = eh12.read_params_range(param_range_file)
        params_in_name = eh12.get_name_params_str()
        params_not_in_Range_dict = ['qq', 'tq']
        
        eh12.set_param(initial_baseline_parameters)
        
        # diff is mut - wild
        # first get baseline data points:
        gv_slope, v_half, top, bottom = cf.calc_act_obj("na12", is_HMM=False)
        self.act_v_half = v_half
        self.act_slope = gv_slope
        ssi_slope, v_half, top, bottom, _ = cf.calc_inact_obj("na12", is_HMM=False)
        self.inact_v_half = v_half
        self.inact_slope = ssi_slope
        self.tau0 = eh12.find_tau0()
        self.per_cur = eh12.find_persistent_current()
        self.peak_amp = eh12.find_peak_amp()
        
        print("debug: " + str(self.peak_amp))
        
        def init_params():
            param_list = []
            print("here are the name, val, min, max of each parameter")
            for param in params_in_name:
                if param not in params_not_in_Range_dict:
                    print(param)
                    val = param_range_dict[param][0]
                    min_bound = param_range_dict[param][1]
                    max_bound = param_range_dict[param][2]
                    print(val)
                    print((min_bound, max_bound))
                    print("")
                    param_list.append(bpop.parameters.Parameter(param, value=val, bounds=(min_bound, max_bound)))
            return param_list

        print("init called")
        self.objectives = []
        self.objectives.append(bpop.objectives.Objective("V_half_Act"))
        self.objectives.append(bpop.objectives.Objective("V_half_inact"))
        self.objectives.append(bpop.objectives.Objective("slope_Act"))
        self.objectives.append(bpop.objectives.Objective("slope_inact"))
        self.objectives.append(bpop.objectives.Objective("tau0"))
        self.objectives.append(bpop.objectives.Objective("pers_curr"))
        self.objectives.append(bpop.objectives.Objective("peak_amp"))
        self.params = init_params()
        
        goal_dict = eh12.read_mutant_protocols(mutant_protocol_csv_name, mutant_name)
        self.V_half_Act_diff_goal = goal_dict['dv_half_act']
        self.V_half_inact_diff_goal = goal_dict['dv_half_ssi']
        # slopes come in the 100 scale since it's a ratio, so we have to divide by 100
        self.slope_Act_ratio_goal = goal_dict['gv_slope']/100
        self.slope_inact_ratio_goal = goal_dict['ssi_slope']/100
        self.tau0_ratio_goal = goal_dict['tau0']/100
        self.per_cur_ratio_goal = goal_dict['persistent']/100
        self.peak_amp_ratio_goal = peak_amp_dict[mutant_name]
        
        print("\n\n\nhere are the goals:")
        print(self.V_half_Act_diff_goal)
        print(self.V_half_inact_diff_goal)
        print(self.slope_Act_ratio_goal)
        print(self.slope_inact_ratio_goal)
        print(self.tau0_ratio_goal)
        print(self.per_cur_ratio_goal)
        print(self.peak_amp_ratio_goal)
        
    def evaluate_with_lists(self, param_values=[]):
        
        print("evaluate_with_lists is called")
        assert len(param_values) == len(self.params), 'no, they have to be equal...'
        
        currh = ggsd.Activation(channel_name = 'na12').h
        currh.sh_na12 = param_values[0]
        currh.tha_na12 = param_values[1]
        currh.qa_na12 = param_values[2]
        currh.Ra_na12 = param_values[3]
        currh.Rb_na12 = param_values[4]
        currh.thi1_na12 = param_values[5]
        currh.thi2_na12 = param_values[6]
        currh.qd_na12 = param_values[7]
        currh.qg_na12 = param_values[8]
        currh.mmin_na12 = param_values[9]
        currh.hmin_na12 = param_values[10]
        currh.q10_na12 = param_values[11]
        currh.Rg_na12 = param_values[12]
        currh.Rd_na12 = param_values[13]
        currh.thinf_na12 = param_values[14]
        currh.qinf_na12 = param_values[15]
        currh.vhalfs_na12 = param_values[16]
        currh.a0s_na12 = param_values[17]
        currh.zetas_na12 = param_values[18]
        currh.gms_na12 = param_values[19]
        currh.smax_na12 = param_values[20]
        currh.vvh_na12 = param_values[21]
        currh.vvs_na12 = param_values[22]
        currh.Ena_na12 = param_values[23]
        currh.Ena_na12 = 55
        
        
        try:
            gv_slope, act_v_half, act_top, act_bottom = cf.calc_act_obj("na12", is_HMM=False)
            ssi_slope, inact_v_half, inact_top, inact_bottom, tau999 = cf.calc_inact_obj("na12", is_HMM=False)
            tau0 = eh12.find_tau0()
            per_cur = eh12.find_persistent_current()
        except:
            return [9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999]

        V_half_Act_diff = act_v_half - self.act_v_half
        V_half_inact_diff = inact_v_half - self.inact_v_half
        gv_slope_ratio = gv_slope/self.act_slope
        ssi_slope_ratio = ssi_slope/self.inact_slope
        tau0_ratio = tau0/self.tau0
        per_cur_ratio = per_cur/self.per_cur
        
        try:
            # eliminate outliers
            act = ggsd.Activation(channel_name = 'na12')
            act.genActivation()
            norm_act_y_val = sorted(list(act.gnorm_vec))
            act_fitted = eh12.get_fitted_act_conductance_arr(act.v_vec, gv_slope, act_v_half, act_top, act_bottom)

            inact = ggsd.Inactivation(channel_name = 'na12')
            inact.genInactivation()
            norm_inact_y_val = sorted(list(inact.inorm_vec))
            inac_fitted = eh12.get_fitted_inact_current_arr(inact.v_vec, ssi_slope, inact_v_half, inact_top, inact_bottom)
        except:
            return [9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999]
        
        try: 
            mutant_peak_amp = eh12.find_peak_amp()
            peak_amp_ratio = mutant_peak_amp/self.peak_amp

            print("debug: " + str(peak_amp_ratio))
        except:
            return [9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999, 9999999999999999]

        
        if self.scaled:            
            return [(V_half_Act_diff/self.V_half_Act_diff_goal - 1)**2 * 1000,
                   (V_half_inact_diff/self.V_half_inact_diff_goal - 1)**2 * 1000,
                   (gv_slope_ratio/self.slope_Act_ratio_goal - 1)**2 * 1000,
                   (ssi_slope_ratio/self.slope_inact_ratio_goal - 1)**2 * 1000,
                   (tau0_ratio/self.tau0_ratio_goal - 1)**2 * 1000,
                   (per_cur_ratio/self.per_cur_ratio_goal - 1)**2 * 1000,
                   (peak_amp_ratio/self.peak_amp_ratio_goal - 1)**2 * 1000]
        else:
            return [(V_half_Act_diff - self.V_half_Act_diff_goal)**2,
                   (V_half_inact_diff - self.V_half_inact_diff_goal)**2,
                   (gv_slope_ratio - self.slope_Act_ratio_goal)**2,
                   (ssi_slope_ratio - self.slope_inact_ratio_goal)**2,
                   (tau0_ratio - self.tau0_ratio_goal)**2,
                   (per_cur_ratio - self.per_cur_ratio_goal)**2,
                   (peak_amp_ratio - self.peak_amp_ratio_goal)**2]
        




evaluator = Vclamp_evaluator(scaled = False)

     
cur_log_file = output_log_file_name

gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = algo._update_history_and_hof
def my_update(halloffame, history, population):
    global gen_counter,cp_freq
    if halloffame is not None:
        halloffame.update(population)
    
    if halloffame:
        best_indvs.append(halloffame[0])
        print(halloffame[0])
        f = open(cur_log_file, 'a')
        f.write(str(halloffame[0]) + '\n')
        f.close()
        #eh12.make_act_plots(halloffame[0])
        #eh12.make_inact_plots(halloffame[0])
    gen_counter = gen_counter+1
    print("Current generation: ", gen_counter)
    if gen_counter%cp_freq == 0:
        fn = '.pkl'
        save_logs(fn,best_indvs,population)

def my_record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)
    f = open(cur_log_file, 'a')
    f.write(str(logbook) + '\n\n\n')
    f.close()
    print('log: \n', logbook, '\n')
    output = open("log.pkl", 'wb')
    pickle.dump(logbook, output)
    output.close()

def save_logs(fn, best_indvs, hof):
    output = open("indv"+fn, 'wb')
    pickle.dump(best_indvs, output)
    output.close()
    output = open("hof"+fn, 'wb')
    pickle.dump(hof, output)

    
#hof = tools.HallOfFame(1, similar=np.array_equal)
hof = tools.ParetoFront()
algo._update_history_and_hof = my_update
algo._record_stats = my_record_stats
pool = multiprocessing.Pool(processes=64)
deap_opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, hof = hof, map_function=pool.map)
#, map_function=pool.map
#deap_opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=5, hof = hof)
cp_file = './cp.pkl'


start_time = time.time()
pop, hof, log, hst = deap_opt.run(max_ngen=num_generations, cp_filename=None)
end_time = time.time()
print(end_time - start_time)