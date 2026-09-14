#!/bin/bash
#SBATCH --job-name=comp3710-unet-testeval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=unet_testeval_%j.out
#SBATCH --error=unet_testeval_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate torch

python -u evaluate_test_set.py "$@"

echo "Job finished $(date)"