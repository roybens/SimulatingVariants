import numpy as np
import time
import generalized_genSim_tel_aviv as ggsd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo
import vclamp_evaluator_relative as vcl_ev
import pickle
from deap import tools
import multiprocessing
import sys

evaluator = vcl_ev.Vclamp_evaluator_relative('./param_stats_narrow.csv', 'A427D')

hof = tools.HallOfFame(1, similar=np.array_equal)
pool = multiprocessing.Pool(processes=64)
deap_opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=100, hof=hof, map_function=pool.map)
cp_file = './cp.pkl'

start_time = time.time()
pop, hof, log, hst = deap_opt.run(max_ngen=5, cp_filename=None)
end_time = time.time()
print("Time elapsed:")
print(end_time - start_time)

print(hof)

print(log)
