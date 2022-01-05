#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J Optimization
#SBATCH --mail-user=jinan@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 0:10:00
#SBATCH --image=balewski/ubu18-py3-mpich:v2

#OpenMP settings:
export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 64 --cpu_bind=cores shifter ./drive.sh

(chmod a+x)
drive.sh:

#!/bin/bash
PATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/bin:$PATH
PYTHONPATH=/global/cscratch1/sd/adisaran/neuronBBP_build2/nrn/lib/python/
nrnivmodl mechs

python3 /global/u1/m/mikelam/SimulatingVariants/Optimization_HHtoHMM.py 

