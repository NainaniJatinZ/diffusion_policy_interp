#!/bin/bash
#SBATCH -c 4                  # Number of Cores per Task
#SBATCH --mem=100GB           # Requested Memory
#SBATCH -p gpu-preempt        # Partition
#SBATCH -G 1                  # Number of GPUs
#SBATCH -t 16:00:00            # Job time limit
#SBATCH -o logs/slurm-%j.out  # %j = job ID for output log
#SBATCH -e logs/slurm-%j.err  # %j = job ID for error log
#SBATCH --constraint=a100     # Constraint to use A100 GPU
#SBATCH -A pi_hzhang2_umass_edu

module load conda/latest
conda activate finetuning 
python sae_analysis/dashboard_gen.py

