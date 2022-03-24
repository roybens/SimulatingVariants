#!/bin/bash
PATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/bin:$PATH
PYTHONPATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/lib/python/
# echo $PYTHONPATH
nrnivmodl mechs
# pip3 install scipy
# pip3 install deap
# pip3 install bluepyopt
python3 /global/u1/m/mikelam/SimulatingVariants/Optimization_HHtoHMM.py 


