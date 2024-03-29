import numpy as np
import time
import generalized_genSim_shorten_time as ggsd
from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import optimize, stats
import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo
import vclamp_evaluator_relative as vcl_ev
import pickle
import time
import numpy as np
from deap import tools
import random
from deap import base, creator
import multiprocessing
import eval_helper as eh
import scoring_functions_relative as sf

evaluator = vcl_ev.Vclamp_evaluator_relative('./param_stats_narrow.csv', 'A427D')

gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = algo._update_history_and_hof
def my_update(halloffame, history, population):
    global gen_counter,cp_freq
    #old_update(halloffame, history, population)
    if halloffame is not None:
        halloffame.update(population)
    #print('hof: ' + str(halloffame))
    #print('population: ' + str(population))
    
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
deap_opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=100, hof = hof, map_function=pool.map)
cp_file = './cp.pkl'

start_time = time.time()
pop, hof, log, hst = deap_opt.run(max_ngen=50, cp_filename=cp_file)
end_time = time.time()
print(end_time - start_time)

print(best_indvs)

print(log)
