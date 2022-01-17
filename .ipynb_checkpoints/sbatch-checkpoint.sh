#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J Optimization
#SBATCH --mail-user=mikelam.us@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 0:10:00
#SBATCH --image=balewski/ubu18-py3-mpich:v2

#OpenMP settings:
export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 1 -c 64 --cpu_bind=cores shifter ./drive.sh

