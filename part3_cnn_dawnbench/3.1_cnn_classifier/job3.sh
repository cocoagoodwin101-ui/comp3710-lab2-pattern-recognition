#!/bin/bash
#SBATCH --job-name=comp3710-part3
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=part3_%j.out
#SBATCH --error=part3_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

python lab2_part3.1_cnn.py
