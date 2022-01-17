#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J Optimization
#SBATCH --mail-user=mikelam.us@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 30:00:00

python3 /global/u1/m/mikelam/SimulatingVariants/Optimization_HHtoHMM.py
