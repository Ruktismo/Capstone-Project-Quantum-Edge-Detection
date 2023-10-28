#!/bin/bash
#SBATCH -J Training-Data-Processing       # Job Name
#SBATCH -p general                        # partition
#SBATCH -q debug                          # QOS
#SBATCH -N 1                              # number of nodes
#SBATCH -c 16                             # Grab entire node (use -c CORE_COUNT if whole node is not needed)
#SBATCH -t 0-00:15:00                     # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out                   # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err                   # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL                   # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu            # who to send the emails to

# mamba create -n QED pillow qiskit matplotlib pylatexenc qiskit-ibm-runtime IPython qiskit-machine-learning qiskit-aer

# move into project directory
cd ~/Capstone-Project-Quantum-Edge-Detection/Pre-Proccessed_Images

# load mamba env
source activate QED_Training

# run code
python Staging.py 1