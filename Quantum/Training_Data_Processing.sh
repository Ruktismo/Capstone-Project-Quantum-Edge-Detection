#!/bin/bash
#SBATCH -J Training-Data-Processing       # Job Name
#SBATCH -p general                        # partition
#SBATCH -q debug                          # QOS
#SBATCH -N 1                              # number of nodes
#SBATCH -c 64                             # Grab entire node (use -c CORE_COUNT if whole node is not needed)
#BATCH -t 0-00:15:00                      # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out                   # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err                   # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL                   # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu            # who to send the emails to

# it can be treated like a normal bash script form here, job will end when this script does.
cd ~/Capstone-Project-Quantum-Edge-Detection    # move into project directory
git pull                                        # pull the latest version from github
cd ./Pre-Proccessed_Images
python Staging.py 3                             # run code