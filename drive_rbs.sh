#!/bin/bash
#PATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/bin:$PATH
#PYTHONPATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/lib/python/
#echo $PYTHONPATH
shifter bash
nrnivmodl mechs
pip3 install scipy
pip3 install deap
pip3 install bluepyopt
python3 //global/homes/r/roybens/SimulatingVariants/Optimization_HHtoHMM.py 


