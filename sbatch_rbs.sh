#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J Optimizationhmm
#SBATCH --mail-user=bens.roy@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH --image=balewski/ubu18-py3-mpich:v5
#SBATCH --output /global/homes/r/roybens/SimulatingVariants/slurm/%A.out
#SBATCH --error /global/homes/r/roybens/SimulatingVariants/slurm/%A.err


#OpenMP settings:
export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#pip install scipy
srun -n 1 -c 64 --cpu_bind=cores ./drive.sh


