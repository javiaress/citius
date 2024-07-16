#!/bin/bash
#SBATCH --job-name=prueba            # Job name
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=1            # -c Run 1 processor per task       
#SBATCH --mem=1gb                    # Job memory request
#SBATCH --gres=gpu:1                 # Request one GPU per job
#SBATCH --time=00:05:00              # Time limit hrs:min:sec
#SBATCH --err errors.err
#SBATCH --out output.out

python prueba.py
