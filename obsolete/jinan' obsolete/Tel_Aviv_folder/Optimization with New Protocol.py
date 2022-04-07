import numpy as np
import time
import generalized_genSim_tel_aviv as ggsd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo
import vclamp_evaluator_relative as vcl_ev
import pickle
import time
from deap import tools
import multiprocessing
import sys

def run(objective_names):
    evaluator = vcl_ev.Vclamp_evaluator_relative('./param_stats.csv', 'na16')
    gen_counter = 0
    best_indvs = []
    cp_freq = 1
    old_update = algo._update_history_and_hof
    def my_update(halloffame, history, population):
        nonlocal gen_counter,cp_freq
        if halloffame is not None:
            halloffame.update(population)

        if halloffame:
            best_indvs.append(halloffame[0])
        gen_counter = gen_counter+1
        print("Current generation: ", gen_counter)
        if gen_counter%cp_freq == 0:
            fn = '.pkl'
            save_logs(fn,best_indvs,population)

    def my_record_stats(stats, logbook, gen, population, invalid_count):
        '''Update the statistics with the new population'''
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=invalid_count, **record)
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

    hof = tools.ParetoFront()
    algo._update_history_and_hof = my_update
    algo._record_stats = my_record_stats
    pool = multiprocessing.Pool(processes=64)
    deap_opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=2, hof = hof, map_function=pool.map)
    cp_file = './cp.pkl'

    start_time = time.time()
    pop, hof, log, hst = deap_opt.run(max_ngen=2, cp_filename=cp_file)
    end_time = time.time()
    print(end_time - start_time)

    print(best_indvs)

    print(log)

if __name__ == "__main__":
    run(sys.argv)