#!/bin/bash
#SBATCH -J QED
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -N 1            # number of nodes
#SBATCH --ecxlusive     # Grab entire node (use -c CORE_COUNT if whole node is not needed)
#SBATCH -t 0-05:00:00   # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu
