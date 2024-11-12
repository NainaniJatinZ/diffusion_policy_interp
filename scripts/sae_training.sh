#!/bin/bash
#SBATCH -c 4                  # Number of Cores per Task
#SBATCH --mem=100GB           # Requested Memory
#SBATCH -p gpu-preempt        # Partition
#SBATCH -G 1                  # Number of GPUs
#SBATCH -t 8:00:00            # Job time limit
#SBATCH -o logs/slurm-%j.out  # %j = job ID for output log
#SBATCH -e logs/slurm-%j.err  # %j = job ID for error log
#SBATCH --constraint=a100     # Constraint to use A100 GPU


module load conda/latest
conda activate finetuning 

# Define values for k and d_hidden to iterate over
ks=(32 64 128)
d_hiddens=(2048 4096)
auxks=(64 128)
# Iterate over each combination of k and d_hidden
for k in "${ks[@]}"; do
    for d_hidden in "${d_hiddens[@]}"; do
        for auxk in "${auxks[@]}"; do 
            python training.py --k $k --d-hidden $d_hidden --auxk $auxk --dead-steps-threshold 200
    
        done
    done
done

