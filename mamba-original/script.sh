#!/bin/bash
#SBATCH --job-name=mamba            # Job name
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=1            # -c Run 1 processor per task       
#SBATCH --mem=100gb                    # Job memory request
#SBATCH --gres=gpu:1                 # Request one GPU per job
#SBATCH --time=05:30:00              # Time limit hrs:min:sec
#SBATCH --err errors.err
#SBATCH --out output_BPI_Challenge_2012_v2.out

python mamba.py
